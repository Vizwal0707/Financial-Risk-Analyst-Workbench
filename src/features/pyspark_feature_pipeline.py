"""PySpark feature engineering pipeline module for distributed batch execution.

Implements identical temporal logic, window specifications, and zero-leakage constraints
as DuckDBFeaturePipeline for distributed Spark clusters or containerized execution.
"""

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def is_pyspark_available() -> bool:
    """Checks whether PySpark package and Java Runtime Environment are present."""
    import os
    from pathlib import Path

    java_home = os.environ.get("JAVA_HOME")
    if not java_home or not Path(java_home).exists():
        return False
    try:
        import pyspark  # noqa: F401

        return True
    except ImportError:
        return False


class PySparkFeaturePipeline:
    """Distributed Spark feature pipeline for large-scale transaction processing.

    Window Specification Equivalence with DuckDB:
    - DuckDB:  RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING
    - PySpark: Window.partitionBy("nameOrig").orderBy("step").rangeBetween(-24, -1)

    Both enforce strict point-in-time correctness: transactions occurring in the current step
    or future steps are strictly excluded from historical rolling metrics.
    """

    def __init__(
        self,
        train_max_step: int = 500,
        val_max_step: int = 620,
        test_max_step: int = 743,
        epsilon: float = 1e-5,
    ) -> None:
        self.train_max_step = train_max_step
        self.val_max_step = val_max_step
        self.test_max_step = test_max_step
        self.epsilon = epsilon

    def transform_spark_df(self, spark_df: Any) -> Any:
        """Transforms a PySpark DataFrame with temporal window specifications."""
        try:
            from pyspark.sql import functions as F
            from pyspark.sql.window import Window
        except ImportError as e:
            raise RuntimeError(
                "PySpark is required to execute transform_spark_df. "
                "Install via 'pip install pyspark' and ensure OpenJDK 11+ is configured."
            ) from e

        # Define backward-looking windows ending at -1 (strictly prior to current step)
        w_orig_1h = Window.partitionBy("nameOrig").orderBy("step").rangeBetween(-1, -1)
        w_orig_6h = Window.partitionBy("nameOrig").orderBy("step").rangeBetween(-6, -1)
        w_orig_24h = Window.partitionBy("nameOrig").orderBy("step").rangeBetween(-24, -1)
        w_orig_72h = Window.partitionBy("nameOrig").orderBy("step").rangeBetween(-72, -1)
        w_dest_24h = Window.partitionBy("nameDest").orderBy("step").rangeBetween(-24, -1)

        # 1. Real-time Attributes & Cyclical Decomposition
        df_feats = (
            spark_df.withColumn("amount_log", F.log1p(F.greatest(F.col("amount"), F.lit(0.0))))
            .withColumn(
                "is_merchant_dest", F.when(F.col("nameDest").startswith("M"), 1).otherwise(0)
            )
            .withColumn("hour_of_day", F.col("step") % 24)
            .withColumn("hour_sin", F.sin(2.0 * math.pi * (F.col("step") % 24) / 24.0))
            .withColumn("hour_cos", F.cos(2.0 * math.pi * (F.col("step") % 24) / 24.0))
            .withColumn("day_of_month", ((F.floor((F.col("step") - 1) / 24.0)) % 31).cast("int"))
            .withColumn("type_CASH_IN", F.when(F.col("type") == "CASH_IN", 1).otherwise(0))
            .withColumn("type_CASH_OUT", F.when(F.col("type") == "CASH_OUT", 1).otherwise(0))
            .withColumn("type_DEBIT", F.when(F.col("type") == "DEBIT", 1).otherwise(0))
            .withColumn("type_PAYMENT", F.when(F.col("type") == "PAYMENT", 1).otherwise(0))
            .withColumn("type_TRANSFER", F.when(F.col("type") == "TRANSFER", 1).otherwise(0))
        )

        # 2. Pre-transaction Solvency Features
        df_feats = (
            df_feats.withColumn(
                "balance_orig_drain_ratio",
                F.col("amount") / (F.col("oldbalanceOrg") + F.lit(self.epsilon)),
            )
            .withColumn(
                "is_orig_zero_balance",
                F.when((F.col("oldbalanceOrg") == 0.0) & (F.col("amount") > 0.0), 1).otherwise(0),
            )
            .withColumn(
                "is_dest_zero_balance",
                F.when((F.col("oldbalanceDest") == 0.0) & (F.col("amount") > 0.0), 1).otherwise(0),
            )
            .withColumn("orig_balance_coverage", F.col("oldbalanceOrg") - F.col("amount"))
            .withColumn(
                "dest_balance_amount_ratio",
                F.col("amount") / (F.col("oldbalanceDest") + F.lit(self.epsilon)),
            )
        )

        # 3. Strict Point-in-Time Rolling Velocity Aggregations
        df_feats = (
            df_feats.withColumn("orig_tx_count_1h", F.count("amount").over(w_orig_1h))
            .withColumn("orig_tx_count_6h", F.count("amount").over(w_orig_6h))
            .withColumn("orig_tx_count_24h", F.count("amount").over(w_orig_24h))
            .withColumn("orig_tx_count_72h", F.count("amount").over(w_orig_72h))
            .withColumn(
                "orig_amount_sum_24h", F.coalesce(F.sum("amount").over(w_orig_24h), F.lit(0.0))
            )
            .withColumn(
                "orig_amount_mean_24h", F.coalesce(F.avg("amount").over(w_orig_24h), F.lit(0.0))
            )
            .withColumn(
                "orig_amount_std_24h",
                F.coalesce(F.stddev_samp("amount").over(w_orig_24h), F.lit(0.0)),
            )
            .withColumn("dest_inflow_count_24h", F.count("amount").over(w_dest_24h))
            .withColumn(
                "dest_inflow_sum_24h",
                F.coalesce(F.sum("amount").over(w_dest_24h), F.lit(0.0)),
            )
        )

        # Derived relative deviations
        df_feats = (
            df_feats.withColumn(
                "orig_amount_deviation_24h",
                F.col("amount") - F.col("orig_amount_mean_24h"),
            )
            .withColumn(
                "orig_amount_zscore_24h",
                (F.col("amount") - F.col("orig_amount_mean_24h"))
                / (F.col("orig_amount_std_24h") + F.lit(self.epsilon)),
            )
            .withColumn(
                "is_first_interaction",
                F.when(
                    (F.col("orig_tx_count_72h") == 0) | (F.col("dest_inflow_count_24h") == 0),
                    1,
                ).otherwise(0),
            )
            .withColumn(
                "partition",
                F.when(F.col("step") <= self.train_max_step, "train")
                .when(F.col("step") <= self.val_max_step, "val")
                .otherwise("test"),
            )
        )

        return df_feats

    @staticmethod
    def emulate_spark_window_parity(df: pd.DataFrame, epsilon: float = 1e-5) -> pd.DataFrame:
        """Pure-Python vectorized emulation of the PySpark Window specification.

        Used to verify exact mathematical parity between PySpark window logic and DuckDB SQL
        in environments where a local Java Runtime Environment is not installed.
        """
        sorted_df = df.sort_values(by=["step"]).copy()
        n = len(sorted_df)

        orig_tx_count_1h = np.zeros(n, dtype=int)
        orig_tx_count_6h = np.zeros(n, dtype=int)
        orig_tx_count_24h = np.zeros(n, dtype=int)
        orig_tx_count_72h = np.zeros(n, dtype=int)
        orig_amount_sum_24h = np.zeros(n, dtype=float)
        orig_amount_mean_24h = np.zeros(n, dtype=float)
        orig_amount_std_24h = np.zeros(n, dtype=float)
        dest_inflow_count_24h = np.zeros(n, dtype=int)
        dest_inflow_sum_24h = np.zeros(n, dtype=float)

        orig_indices = {}
        for idx, orig in enumerate(sorted_df["nameOrig"].values):
            if orig not in orig_indices:
                orig_indices[orig] = []
            orig_indices[orig].append(idx)

        dest_indices = {}
        for idx, dest in enumerate(sorted_df["nameDest"].values):
            if dest not in dest_indices:
                dest_indices[dest] = []
            dest_indices[dest].append(idx)

        steps = sorted_df["step"].values
        amounts = sorted_df["amount"].values

        for idx in range(n):
            s = steps[idx]
            orig = sorted_df["nameOrig"].values[idx]
            dest = sorted_df["nameDest"].values[idx]

            # Sender historical window: strictly preceding steps
            s_idxs = orig_indices[orig]
            # Prior transactions by this sender
            prior_s = [i for i in s_idxs if i < idx and steps[i] < s]

            p_1h = [amounts[i] for i in prior_s if s - 1 <= steps[i] < s]
            p_6h = [amounts[i] for i in prior_s if s - 6 <= steps[i] < s]
            p_24h = [amounts[i] for i in prior_s if s - 24 <= steps[i] < s]
            p_72h = [amounts[i] for i in prior_s if s - 72 <= steps[i] < s]

            orig_tx_count_1h[idx] = len(p_1h)
            orig_tx_count_6h[idx] = len(p_6h)
            orig_tx_count_24h[idx] = len(p_24h)
            orig_tx_count_72h[idx] = len(p_72h)

            if p_24h:
                orig_amount_sum_24h[idx] = sum(p_24h)
                orig_amount_mean_24h[idx] = np.mean(p_24h)
                orig_amount_std_24h[idx] = np.std(p_24h, ddof=1) if len(p_24h) > 1 else 0.0

            # Destination historical window
            d_idxs = dest_indices[dest]
            prior_d = [i for i in d_idxs if i < idx and steps[i] < s]
            p_dest_24h = [amounts[i] for i in prior_d if s - 24 <= steps[i] < s]

            dest_inflow_count_24h[idx] = len(p_dest_24h)
            if p_dest_24h:
                dest_inflow_sum_24h[idx] = sum(p_dest_24h)

        out = pd.DataFrame(index=sorted_df.index)
        out["orig_tx_count_1h"] = orig_tx_count_1h
        out["orig_tx_count_6h"] = orig_tx_count_6h
        out["orig_tx_count_24h"] = orig_tx_count_24h
        out["orig_tx_count_72h"] = orig_tx_count_72h
        out["orig_amount_sum_24h"] = orig_amount_sum_24h
        out["orig_amount_mean_24h"] = orig_amount_mean_24h
        out["orig_amount_std_24h"] = orig_amount_std_24h
        out["dest_inflow_count_24h"] = dest_inflow_count_24h
        out["dest_inflow_sum_24h"] = dest_inflow_sum_24h
        return out

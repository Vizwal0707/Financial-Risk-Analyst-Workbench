"""High-throughput vectorized DuckDB point-in-time feature engineering pipeline."""

import logging
import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from src.ingestion.schemas import verify_zero_leakage

logger = logging.getLogger(__name__)


class DuckDBFeaturePipeline:
    """Vectorized SQL feature engineering backend operating directly on Parquet files.

    Guarantees strict point-in-time correctness by computing rolling historical windows
    exclusively over preceding discrete simulation steps (step < current step).
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

    def get_feature_sql(self, source_path: str, limit: int | None = None) -> str:
        """Constructs the complete vectorized feature engineering SQL query."""
        path_str = str(Path(source_path).resolve()).replace("'", "''")
        limit_clause = f"LIMIT {limit}" if limit is not None else ""

        sql = f"""
        WITH raw_source AS (
            SELECT
                step,
                type,
                amount,
                nameOrig,
                oldbalanceOrg,
                nameDest,
                oldbalanceDest,
                isFraud
            FROM read_parquet('{path_str}')
            {limit_clause}
        ),
        base_features AS (
            SELECT
                step,
                type,
                amount,
                nameOrig,
                oldbalanceOrg,
                nameDest,
                oldbalanceDest,
                isFraud,

                -- Partition allocation (Step-atomic guarantee)
                CASE
                    WHEN step <= {self.train_max_step} THEN 'train'
                    WHEN step <= {self.val_max_step} THEN 'val'
                    ELSE 'test'
                END AS partition,

                -- 1. Real-time Transaction Attributes
                ln(amount + 1.0) AS amount_log,
                CASE WHEN nameDest LIKE 'M%' THEN 1 ELSE 0 END AS is_merchant_dest,
                (step % 24) AS hour_of_day,
                sin(2.0 * pi() * (step % 24) / 24.0) AS hour_sin,
                cos(2.0 * pi() * (step % 24) / 24.0) AS hour_cos,
                CAST(floor((step - 1) / 24.0) AS INTEGER) % 31 AS day_of_month,

                -- Corridor One-Hot Flags
                CASE WHEN type = 'CASH_IN' THEN 1 ELSE 0 END AS type_CASH_IN,
                CASE WHEN type = 'CASH_OUT' THEN 1 ELSE 0 END AS type_CASH_OUT,
                CASE WHEN type = 'DEBIT' THEN 1 ELSE 0 END AS type_DEBIT,
                CASE WHEN type = 'PAYMENT' THEN 1 ELSE 0 END AS type_PAYMENT,
                CASE WHEN type = 'TRANSFER' THEN 1 ELSE 0 END AS type_TRANSFER,

                -- 2. Pre-transaction Solvency & Drain Dynamics (Zero Post-Balance Leakage)
                amount / (oldbalanceOrg + {self.epsilon}) AS balance_orig_drain_ratio,
                CASE WHEN oldbalanceOrg = 0.0 AND amount > 0.0 THEN 1 ELSE 0 END AS is_orig_zero_balance,
                CASE WHEN oldbalanceDest = 0.0 AND amount > 0.0 THEN 1 ELSE 0 END AS is_dest_zero_balance,
                (oldbalanceOrg - amount) AS orig_balance_coverage,
                amount / (oldbalanceDest + {self.epsilon}) AS dest_balance_amount_ratio,

                -- 3. Strict Point-in-Time Rolling Velocity & Historical Windows (step < current step)
                COUNT(*) OVER w_orig_1h AS orig_tx_count_1h,
                COUNT(*) OVER w_orig_6h AS orig_tx_count_6h,
                COUNT(*) OVER w_orig_24h AS orig_tx_count_24h,
                COUNT(*) OVER w_orig_72h AS orig_tx_count_72h,
                COALESCE(SUM(amount) OVER w_orig_24h, 0.0) AS orig_amount_sum_24h,
                COALESCE(AVG(amount) OVER w_orig_24h, 0.0) AS orig_amount_mean_24h,
                COALESCE(STDDEV_SAMP(amount) OVER w_orig_24h, 0.0) AS orig_amount_std_24h,

                COUNT(*) OVER w_dest_24h AS dest_inflow_count_24h,
                COALESCE(SUM(amount) OVER w_dest_24h, 0.0) AS dest_inflow_sum_24h

            FROM raw_source
            WINDOW
                w_orig_1h AS (PARTITION BY nameOrig ORDER BY step RANGE BETWEEN 1 PRECEDING AND 1 PRECEDING),
                w_orig_6h AS (PARTITION BY nameOrig ORDER BY step RANGE BETWEEN 6 PRECEDING AND 1 PRECEDING),
                w_orig_24h AS (PARTITION BY nameOrig ORDER BY step RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING),
                w_orig_72h AS (PARTITION BY nameOrig ORDER BY step RANGE BETWEEN 72 PRECEDING AND 1 PRECEDING),
                w_dest_24h AS (PARTITION BY nameDest ORDER BY step RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING)
        )
        SELECT
            *,
            amount - orig_amount_mean_24h AS orig_amount_deviation_24h,
            (amount - orig_amount_mean_24h) / (orig_amount_std_24h + {self.epsilon}) AS orig_amount_zscore_24h,
            CASE WHEN orig_tx_count_72h = 0 OR dest_inflow_count_24h = 0 THEN 1 ELSE 0 END AS is_first_interaction
        FROM base_features
        """
        return sql

    def transform_df(self, source_path: str, limit: int | None = None) -> pd.DataFrame:
        """Executes feature engineering and returns resulting DataFrame."""
        conn = duckdb.connect()
        sql = self.get_feature_sql(source_path=source_path, limit=limit)
        df = conn.execute(sql).df()

        # Enforce zero-leakage assertions
        feature_cols = set(df.columns)
        verify_zero_leakage(feature_cols)

        return df

    def process_and_save(
        self,
        source_parquet_path: str,
        output_dir: str = "data/processed",
        save_partitions: bool = True,
        compression: str = "snappy",
    ) -> dict[str, Any]:
        """Processes source Parquet directly via DuckDB COPY into compressed destination Parquet files."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initiating DuckDB high-throughput feature generation on: {source_parquet_path}"
        )
        start_time = time.perf_counter()

        conn = duckdb.connect()
        # Set generous memory limit and multi-threaded worker allocation
        conn.execute("SET memory_limit = '6GB'")
        conn.execute("SET threads = 4")

        base_sql = self.get_feature_sql(source_path=source_parquet_path)
        unified_output_file = out_dir / "paysim_featurized.parquet"

        # 1. Export unified featurized dataset
        copy_sql = f"""
        COPY (
            {base_sql}
        ) TO '{str(unified_output_file.resolve())}' (FORMAT PARQUET, COMPRESSION {compression});
        """
        logger.info("Executing vectorized feature computation & Parquet export...")
        conn.execute(copy_sql)

        elapsed_seconds = time.perf_counter() - start_time
        file_size_mb = unified_output_file.stat().st_size / (1024 * 1024)

        # 2. Extract partition counts and summary statistics
        summary = conn.execute(f"""
        SELECT
            partition,
            COUNT(*) AS row_count,
            SUM(isFraud) AS fraud_count,
            MIN(step) AS min_step,
            MAX(step) AS max_step
        FROM read_parquet('{str(unified_output_file.resolve())}')
        GROUP BY partition
        ORDER BY min_step
        """).df()

        total_rows = int(summary["row_count"].sum())
        total_fraud = int(summary["fraud_count"].sum())
        throughput_rows_sec = total_rows / elapsed_seconds if elapsed_seconds > 0 else 0

        # 3. Save discrete train, val, and test partitions if requested
        if save_partitions:
            logger.info("Materializing discrete partition Parquet splits...")
            for part in ["train", "val", "test"]:
                part_path = out_dir / f"{part}.parquet"
                conn.execute(f"""
                COPY (
                    SELECT * EXCLUDE (partition)
                    FROM read_parquet('{str(unified_output_file.resolve())}')
                    WHERE partition = '{part}'
                ) TO '{str(part_path.resolve())}' (FORMAT PARQUET, COMPRESSION {compression});
                """)

        logger.info(
            f"Feature engineering completed in {elapsed_seconds:.2f}s "
            f"({throughput_rows_sec:,.0f} rows/s, {total_rows:,} rows, {file_size_mb:.1f} MB)."
        )

        return {
            "total_records": total_rows,
            "total_fraud": total_fraud,
            "elapsed_seconds": round(elapsed_seconds, 2),
            "throughput_rows_sec": round(throughput_rows_sec, 1),
            "output_size_mb": round(file_size_mb, 2),
            "partitions": summary.to_dict(orient="records"),
        }

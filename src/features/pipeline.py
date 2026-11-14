"""Unified feature engineering pipeline manager, scaler fitting, and inference transformation."""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.features.balance_transforms import BalanceTransformer
from src.features.duckdb_feature_pipeline import DuckDBFeaturePipeline
from src.features.transaction_transforms import TransactionTransformer
from src.ingestion.schemas import verify_zero_leakage

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_COLUMNS = [
    "amount",
    "amount_log",
    "oldbalanceOrg",
    "oldbalanceDest",
    "balance_orig_drain_ratio",
    "is_orig_zero_balance",
    "is_dest_zero_balance",
    "orig_balance_coverage",
    "dest_balance_amount_ratio",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "day_of_month",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "orig_tx_count_1h",
    "orig_tx_count_6h",
    "orig_tx_count_24h",
    "orig_tx_count_72h",
    "orig_amount_sum_24h",
    "orig_amount_mean_24h",
    "orig_amount_std_24h",
    "orig_amount_deviation_24h",
    "orig_amount_zscore_24h",
    "dest_inflow_count_24h",
    "dest_inflow_sum_24h",
    "is_first_interaction",
]


class FinancialFeaturePipeline:
    """Orchestrates end-to-end feature extraction, training-set scaler fitting, and inference encoding."""

    def __init__(
        self,
        scaler_type: str = "robust",
        feature_columns: list[str] | None = None,
        train_max_step: int = 500,
        val_max_step: int = 620,
        test_max_step: int = 743,
    ) -> None:
        self.feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
        self.scaler = RobustScaler() if scaler_type == "robust" else None
        self.is_fitted = False
        self.train_max_step = train_max_step
        self.val_max_step = val_max_step
        self.test_max_step = test_max_step

        self.tx_transformer = TransactionTransformer()
        self.balance_transformer = BalanceTransformer()
        self.duckdb_pipeline = DuckDBFeaturePipeline(
            train_max_step=train_max_step,
            val_max_step=val_max_step,
            test_max_step=test_max_step,
        )

    def fit_scalers(self, train_features_df: pd.DataFrame) -> "FinancialFeaturePipeline":
        """Fits scaler strictly on training partition features to prevent distribution leakage."""
        logger.info(
            f"Fitting {type(self.scaler).__name__} strictly on {len(train_features_df):,} training records."
        )
        X_train = train_features_df[self.feature_columns].values
        if self.scaler is not None:
            self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def scale_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Scales feature DataFrame using fitted training parameters."""
        if not self.is_fitted:
            raise RuntimeError(
                "Pipeline must be fitted on training partition before scaling features."
            )
        X = features_df[self.feature_columns].values
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X

    def transform_single_inference(
        self,
        transaction_dict: dict[str, Any],
        historical_stats: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Transforms a single live transaction payload into model feature vector at inference time.

        historical_stats provides pre-aggregated point-in-time metrics from cache/redis.
        """
        # Validate that no forbidden leakage columns were submitted in payload
        verify_zero_leakage(set(transaction_dict.keys()))

        df = pd.DataFrame([transaction_dict])

        # 1. Real-time attributes & balance transforms
        tx_feats = self.tx_transformer.transform(df)
        bal_feats = self.balance_transformer.transform(df)

        combined = pd.concat([tx_feats, bal_feats], axis=1)

        # 2. Inject historical velocity stats (default to 0 if cold entity)
        h = historical_stats or {}
        combined["orig_tx_count_1h"] = h.get("orig_tx_count_1h", 0)
        combined["orig_tx_count_6h"] = h.get("orig_tx_count_6h", 0)
        combined["orig_tx_count_24h"] = h.get("orig_tx_count_24h", 0)
        combined["orig_tx_count_72h"] = h.get("orig_tx_count_72h", 0)
        combined["orig_amount_sum_24h"] = h.get("orig_amount_sum_24h", 0.0)
        combined["orig_amount_mean_24h"] = h.get("orig_amount_mean_24h", 0.0)
        combined["orig_amount_std_24h"] = h.get("orig_amount_std_24h", 0.0)
        combined["dest_inflow_count_24h"] = h.get("dest_inflow_count_24h", 0)
        combined["dest_inflow_sum_24h"] = h.get("dest_inflow_sum_24h", 0.0)

        # Derived relative deviations
        amt = combined["amount"].values[0]
        mean_amt = combined["orig_amount_mean_24h"].values[0]
        std_amt = combined["orig_amount_std_24h"].values[0]

        combined["orig_amount_deviation_24h"] = amt - mean_amt
        combined["orig_amount_zscore_24h"] = (amt - mean_amt) / (std_amt + 1e-5)
        combined["is_first_interaction"] = int(
            combined["orig_tx_count_72h"].values[0] == 0
            or combined["dest_inflow_count_24h"].values[0] == 0
        )

        return combined[self.feature_columns]

    def save_state(self, filepath: str) -> None:
        """Serializes fitted pipeline state to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Serialized feature pipeline state to: {filepath}")

    @classmethod
    def load_state(cls, filepath: str) -> "FinancialFeaturePipeline":
        """Loads fitted pipeline state from disk."""
        return joblib.load(filepath)

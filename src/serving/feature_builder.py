"""Point-in-time feature reconstruction and strict anti-leakage validation for real-time inference."""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.schemas.transaction import (
    FORBIDDEN_LEAKAGE_COLUMNS,
    HistoricalContext,
    TransactionInput,
)

logger = logging.getLogger(__name__)

# Audited 31-feature roster matching models/xgboost_best.joblib in exact order
MODEL_FEATURE_NAMES: List[str] = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_log",
    "is_merchant_dest",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "day_of_month",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "balance_orig_drain_ratio",
    "is_orig_zero_balance",
    "is_dest_zero_balance",
    "orig_balance_coverage",
    "dest_balance_amount_ratio",
    "orig_tx_count_1h",
    "orig_tx_count_6h",
    "orig_tx_count_24h",
    "orig_tx_count_72h",
    "orig_amount_sum_24h",
    "orig_amount_mean_24h",
    "orig_amount_std_24h",
    "dest_inflow_count_24h",
    "dest_inflow_sum_24h",
    "orig_amount_deviation_24h",
    "orig_amount_zscore_24h",
    "is_first_interaction",
]


class MissingFeatureContextError(ValueError):
    """Raised when historical entity context is missing and cold-start fallback is not explicitly permitted."""
    pass


class ForbiddenLeakageError(ValueError):
    """Raised when payload contains forbidden post-transaction or target leakage fields."""
    pass


class FeatureBuilder:
    """Constructs point-in-time model feature vectors for live transactions at decision time.

    Enforces:
    1. Zero data leakage (forbids target isFraud, newbalanceOrig, newbalanceDest, isFlaggedFraud).
    2. Exact mathematical parity with DuckDBFeaturePipeline offline training transformations.
    3. Strict feature names, types, and ordering matching the trained XGBoost model.
    4. Guard against silent zero-filling: requires explicit historical context or verified cold-start authorization.
    """

    def __init__(self, epsilon: float = 1e-5) -> None:
        self.epsilon = epsilon
        self.feature_names = list(MODEL_FEATURE_NAMES)

    def validate_payload(self, raw_data: Dict[str, Any]) -> None:
        """Verifies that no forbidden leakage fields exist in the input dictionary."""
        forbidden = set(raw_data.keys()).intersection(FORBIDDEN_LEAKAGE_COLUMNS)
        if forbidden:
            raise ForbiddenLeakageError(
                f"Data leakage guard triggered! Forbidden post-transaction columns detected: {sorted(forbidden)}. "
                f"These fields represent future or outcome information unavailable at decision time."
            )

    def transform_single(
        self,
        tx: TransactionInput,
    ) -> Tuple[np.ndarray, bool]:
        """Transforms a single live transaction into an exact (1, 31) float64 feature matrix.

        Returns:
            Tuple of (feature_matrix_1x31, cold_start_applied_bool)
        """
        feats, cold_start = self._build_features_dict(tx)
        matrix = np.array([[feats[col] for col in self.feature_names]], dtype=np.float64)
        return matrix, cold_start

    def transform_batch(
        self,
        transactions: Sequence[TransactionInput],
    ) -> Tuple[np.ndarray, List[bool]]:
        """Vectorized transformation of multiple transactions into an (N, 31) feature matrix.

        Guarantees 100% numerical parity with single-transaction transformation.
        """
        n = len(transactions)
        if n == 0:
            return np.empty((0, len(self.feature_names)), dtype=np.float64), []

        feature_dicts: List[Dict[str, float]] = []
        cold_start_flags: List[bool] = []

        for tx in transactions:
            feats, cold_start = self._build_features_dict(tx)
            feature_dicts.append(feats)
            cold_start_flags.append(cold_start)

        matrix = np.empty((n, len(self.feature_names)), dtype=np.float64)
        for i, row in enumerate(feature_dicts):
            for j, col in enumerate(self.feature_names):
                matrix[i, j] = row[col]

        return matrix, cold_start_flags

    def transform_dataframe(
        self,
        transactions: Sequence[TransactionInput],
    ) -> Tuple[pd.DataFrame, List[bool]]:
        """Transforms transactions into a named pandas DataFrame for TreeSHAP or inspection."""
        matrix, cold_flags = self.transform_batch(transactions)
        df = pd.DataFrame(matrix, columns=self.feature_names)
        return df, cold_flags

    def _build_features_dict(self, tx: TransactionInput) -> Tuple[Dict[str, float], bool]:
        """Extracts and computes all 31 features for a single transaction input."""
        # 1. Base transaction parameters
        amt = float(tx.amount)
        old_org = float(tx.oldbalanceOrg)
        old_dest = float(tx.oldbalanceDest)
        step = int(tx.step)
        name_dest = str(tx.nameDest)
        corridor = tx.type.upper()

        # 2. Real-time derived features
        amt_log = float(np.log1p(amt))
        is_merchant = 1.0 if name_dest.startswith("M") else 0.0
        hour = step % 24
        hour_sin = float(np.sin(2.0 * np.pi * hour / 24.0))
        hour_cos = float(np.cos(2.0 * np.pi * hour / 24.0))
        day_of_month = float(int(np.floor((step - 1) / 24.0)) % 31)

        # 3. Corridor one-hot encoding
        type_cash_in = 1.0 if corridor == "CASH_IN" else 0.0
        type_cash_out = 1.0 if corridor == "CASH_OUT" else 0.0
        type_debit = 1.0 if corridor == "DEBIT" else 0.0
        type_payment = 1.0 if corridor == "PAYMENT" else 0.0
        type_transfer = 1.0 if corridor == "TRANSFER" else 0.0

        # 4. Solvency & Balance transforms
        drain_ratio = float(amt / (old_org + self.epsilon))
        is_orig_zero = 1.0 if (old_org == 0.0 and amt > 0.0) else 0.0
        is_dest_zero = 1.0 if (old_dest == 0.0 and amt > 0.0) else 0.0
        orig_cov = float(old_org - amt)
        dest_ratio = float(amt / (old_dest + self.epsilon))

        # 5. Point-in-Time Historical velocity & context
        ctx = tx.historical_context
        cold_start_applied = False

        if ctx is not None:
            c_1h = float(ctx.orig_tx_count_1h)
            c_6h = float(ctx.orig_tx_count_6h)
            c_24h = float(ctx.orig_tx_count_24h)
            c_72h = float(ctx.orig_tx_count_72h)
            s_24h = float(ctx.orig_amount_sum_24h)
            m_24h = float(ctx.orig_amount_mean_24h)
            std_24h = float(ctx.orig_amount_std_24h)
            dest_c_24h = float(ctx.dest_inflow_count_24h)
            dest_s_24h = float(ctx.dest_inflow_sum_24h)
            cold_start_applied = False
        else:
            if not tx.allow_cold_start:
                raise MissingFeatureContextError(
                    f"Historical feature context missing for entity {tx.nameOrig}. "
                    f"Point-in-time velocity features (orig_tx_count_1h..72h, orig_amount_mean_24h, etc.) "
                    f"must be retrieved from the operational feature store (e.g., Redis/Feast). "
                    f"To permit verified cold-start evaluation for first-time entities, set 'allow_cold_start=True'."
                )
            # Verified cold-start defaults (PaySim synthetic properties: 0 prior transactions)
            c_1h = 0.0
            c_6h = 0.0
            c_24h = 0.0
            c_72h = 0.0
            s_24h = 0.0
            m_24h = 0.0
            std_24h = 0.0
            dest_c_24h = 0.0
            dest_s_24h = 0.0
            cold_start_applied = True

        # Derived velocity relative dynamics
        amt_dev_24h = float(amt - m_24h)
        amt_zscore_24h = float((amt - m_24h) / (std_24h + self.epsilon))
        is_first_interaction = 1.0 if (c_72h == 0.0 or dest_c_24h == 0.0) else 0.0

        features_dict: Dict[str, float] = {
            "amount": amt,
            "oldbalanceOrg": old_org,
            "oldbalanceDest": old_dest,
            "amount_log": amt_log,
            "is_merchant_dest": is_merchant,
            "hour_of_day": float(hour),
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_month": day_of_month,
            "type_CASH_IN": type_cash_in,
            "type_CASH_OUT": type_cash_out,
            "type_DEBIT": type_debit,
            "type_PAYMENT": type_payment,
            "type_TRANSFER": type_transfer,
            "balance_orig_drain_ratio": drain_ratio,
            "is_orig_zero_balance": is_orig_zero,
            "is_dest_zero_balance": is_dest_zero,
            "orig_balance_coverage": orig_cov,
            "dest_balance_amount_ratio": dest_ratio,
            "orig_tx_count_1h": c_1h,
            "orig_tx_count_6h": c_6h,
            "orig_tx_count_24h": c_24h,
            "orig_tx_count_72h": c_72h,
            "orig_amount_sum_24h": s_24h,
            "orig_amount_mean_24h": m_24h,
            "orig_amount_std_24h": std_24h,
            "dest_inflow_count_24h": dest_c_24h,
            "dest_inflow_sum_24h": dest_s_24h,
            "orig_amount_deviation_24h": amt_dev_24h,
            "orig_amount_zscore_24h": amt_zscore_24h,
            "is_first_interaction": is_first_interaction,
        }

        return features_dict, cold_start_applied


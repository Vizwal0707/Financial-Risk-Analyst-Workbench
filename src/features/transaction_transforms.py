"""Modular transformations for transaction attributes, cyclical temporal indicators, and corridors."""

import numpy as np
import pandas as pd

VALID_CORRIDORS = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


class TransactionTransformer:
    """Computes stateless real-time attributes, log transformations, and cyclical time features."""

    def __init__(self, corridors: list[str] | None = None) -> None:
        self.corridors = sorted(corridors or VALID_CORRIDORS)

    def fit(self, df: pd.DataFrame) -> "TransactionTransformer":
        """Fits any stateful parameters (e.g. known corridor categories)."""
        # For PaySim, corridor types are fixed across all partitions
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies scalar and cyclical transformations to the transaction DataFrame."""
        out = pd.DataFrame(index=df.index)

        # 1. Log-transformed amount (mitigates severe right-skew)
        out["amount"] = df["amount"].astype(float)
        out["amount_log"] = np.log1p(np.maximum(df["amount"].values, 0.0))

        # 2. Destination entity classification
        out["is_merchant_dest"] = df["nameDest"].astype(str).str.startswith("M").astype(int)

        # 3. Cyclical and calendar time decomposition
        step = df["step"].values
        hour = step % 24
        out["hour_of_day"] = hour.astype(int)
        out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        out["day_of_month"] = (((step - 1) // 24) % 31).astype(int)

        # 4. Deterministic one-hot encoding for transaction corridor
        for corridor in self.corridors:
            out[f"type_{corridor}"] = (df["type"] == corridor).astype(int)

        return out

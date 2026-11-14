"""Modular transformations for pre-transaction solvency, account draining, and balance dynamics."""

import pandas as pd


class BalanceTransformer:
    """Computes pre-transaction balance solvency ratios without post-transaction leakage."""

    def __init__(self, epsilon: float = 1e-5) -> None:
        self.epsilon = epsilon

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes solvency and account depletion features strictly using pre-transaction balances."""
        out = pd.DataFrame(index=df.index)

        amount = df["amount"].values.astype(float)
        old_org = df["oldbalanceOrg"].values.astype(float)
        old_dest = df["oldbalanceDest"].values.astype(float)

        # 1. Sender account drain ratio (ratio close to 1.0 indicates total account liquidation)
        out["balance_orig_drain_ratio"] = amount / (old_org + self.epsilon)

        # 2. Structural zero-balance anomalies
        out["is_orig_zero_balance"] = ((old_org == 0.0) & (amount > 0.0)).astype(int)
        out["is_dest_zero_balance"] = ((old_dest == 0.0) & (amount > 0.0)).astype(int)

        # 3. Unmitigated overdraft / balance coverage exposure
        out["orig_balance_coverage"] = old_org - amount

        # 4. Destination balance impact ratio
        out["dest_balance_amount_ratio"] = amount / (old_dest + self.epsilon)

        # 5. Pre-transaction base balances (strictly legitimate inputs)
        out["oldbalanceOrg"] = old_org
        out["oldbalanceDest"] = old_dest

        return out

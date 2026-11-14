"""Pandera validation schemas and data contracts for transaction datasets."""

import pandera.pandas as pa
from pandera.typing import Series
from pydantic import BaseModel, Field, field_validator

VALID_TRANSACTION_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}
FORBIDDEN_LEAKAGE_COLUMNS = {
    "newbalanceOrig",
    "newbalanceDest",
    "isFlaggedFraud",
    "is_flagged_fraud",
}


class RawTransactionSchema(pa.DataFrameModel):
    """Pandera schema contract for raw PaySim benchmark transactions."""

    step: Series[int] = pa.Field(
        ge=1,
        le=744,
        description="Discrete simulation hour step in [1, 744]",
        nullable=False,
    )
    type: Series[str] = pa.Field(
        isin=list(VALID_TRANSACTION_TYPES),
        description="Transaction corridor type",
        nullable=False,
    )
    amount: Series[float] = pa.Field(
        ge=0.0,
        description="Transaction nominal amount",
        nullable=False,
    )
    nameOrig: Series[str] = pa.Field(
        description="Unique customer identifier for sender",
        nullable=False,
    )
    oldbalanceOrg: Series[float] = pa.Field(
        ge=0.0,
        description="Sender balance immediately prior to transaction",
        nullable=False,
    )
    newbalanceOrig: Series[float] = pa.Field(
        ge=0.0,
        description="Sender balance post-transaction (raw data only; forbidden in inference)",
        nullable=False,
    )
    nameDest: Series[str] = pa.Field(
        description="Unique account identifier for recipient",
        nullable=False,
    )
    oldbalanceDest: Series[float] = pa.Field(
        ge=0.0,
        description="Recipient balance immediately prior to transaction",
        nullable=False,
    )
    newbalanceDest: Series[float] = pa.Field(
        ge=0.0,
        description="Recipient balance post-transaction (raw data only; forbidden in inference)",
        nullable=False,
    )
    isFraud: Series[int] = pa.Field(
        isin=[0, 1],
        description="Ground truth binary fraud label (0=Legit, 1=Fraud)",
        nullable=False,
    )
    isFlaggedFraud: Series[int] = pa.Field(
        isin=[0, 1],
        description="Naive heuristic flag (>200k transfer); forbidden in model features",
        nullable=False,
    )

    class Config:
        strict = False
        coerce = True


class FeaturizedTransactionSchema(pa.DataFrameModel):
    """Pandera schema validating point-in-time featurized transactions for model training."""

    step: Series[int] = pa.Field(ge=1, le=744, nullable=False)
    type: Series[str] = pa.Field(isin=list(VALID_TRANSACTION_TYPES), nullable=False)
    amount: Series[float] = pa.Field(ge=0.0, nullable=False)
    amount_log: Series[float] = pa.Field(ge=0.0, nullable=False)
    oldbalanceOrg: Series[float] = pa.Field(ge=0.0, nullable=False)
    oldbalanceDest: Series[float] = pa.Field(ge=0.0, nullable=False)
    balance_orig_drain_ratio: Series[float] = pa.Field(ge=0.0, nullable=False)
    is_orig_zero_balance: Series[int] = pa.Field(isin=[0, 1], nullable=False)
    is_dest_zero_balance: Series[int] = pa.Field(isin=[0, 1], nullable=False)
    orig_balance_coverage: Series[float] = pa.Field(nullable=False)
    dest_balance_amount_ratio: Series[float] = pa.Field(ge=0.0, nullable=False)
    hour_of_day: Series[int] = pa.Field(ge=0, le=23, nullable=False)
    hour_sin: Series[float] = pa.Field(ge=-1.0, le=1.0, nullable=False)
    hour_cos: Series[float] = pa.Field(ge=-1.0, le=1.0, nullable=False)
    day_of_month: Series[int] = pa.Field(ge=0, le=31, nullable=False)
    is_merchant_dest: Series[int] = pa.Field(isin=[0, 1], nullable=False)

    class Config:
        strict = False
        coerce = True


class InferenceTransactionInput(BaseModel):
    """Pydantic model representing a single real-time transaction presented for risk scoring.

    Guarantees zero-leakage: post-transaction balances and rule flags are not accepted.
    """

    step: int = Field(..., ge=1, le=1000, description="Simulation or calendar step (hour)")
    type: str = Field(
        ..., description="Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER"
    )
    amount: float = Field(..., ge=0.0, description="Transaction nominal value")
    nameOrig: str = Field(..., min_length=2, max_length=64, description="Originating account ID")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Sender pre-transaction balance")
    nameDest: str = Field(..., min_length=2, max_length=64, description="Recipient account ID")
    oldbalanceDest: float = Field(..., ge=0.0, description="Recipient pre-transaction balance")

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        upper_val = value.strip().upper()
        if upper_val not in VALID_TRANSACTION_TYPES:
            raise ValueError(
                f"Invalid transaction type '{value}'. Must be one of {sorted(VALID_TRANSACTION_TYPES)}"
            )
        return upper_val


def verify_zero_leakage(column_names: set[str]) -> None:
    """Asserts that no forbidden leakage columns exist in the feature set."""
    intersection = column_names.intersection(FORBIDDEN_LEAKAGE_COLUMNS)
    if intersection:
        raise ValueError(
            f"Target leakage violation detected! Forbidden columns present in feature set: {sorted(intersection)}. "
            f"Post-transaction balances and heuristic rule flags must strictly be excluded."
        )

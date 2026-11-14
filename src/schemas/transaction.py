"""Transaction request validation schemas, data contracts, and anti-leakage guards."""

import uuid
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, field_validator, model_validator

VALID_TRANSACTION_TYPES: Set[str] = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}

FORBIDDEN_LEAKAGE_COLUMNS: Set[str] = {
    "isFraud",
    "is_fraud",
    "newbalanceOrig",
    "newbalance_orig",
    "newbalanceDest",
    "newbalance_dest",
    "isFlaggedFraud",
    "is_flagged_fraud",
}


class HistoricalContext(BaseModel):
    """Precomputed point-in-time velocity and entity interaction metrics.

    Supplied by an operational feature store (e.g., Redis, Feast, Hopsworks)
    reflecting observations strictly prior to the current transaction.
    """

    orig_tx_count_1h: int = Field(default=0, ge=0, description="Origin transactions in prior 1 hour")
    orig_tx_count_6h: int = Field(default=0, ge=0, description="Origin transactions in prior 6 hours")
    orig_tx_count_24h: int = Field(default=0, ge=0, description="Origin transactions in prior 24 hours")
    orig_tx_count_72h: int = Field(default=0, ge=0, description="Origin transactions in prior 72 hours")
    orig_amount_sum_24h: float = Field(default=0.0, ge=0.0, description="Origin transaction sum in prior 24 hours")
    orig_amount_mean_24h: float = Field(default=0.0, ge=0.0, description="Origin mean amount in prior 24 hours")
    orig_amount_std_24h: float = Field(default=0.0, ge=0.0, description="Origin amount standard deviation in prior 24 hours")
    dest_inflow_count_24h: int = Field(default=0, ge=0, description="Destination inbound transactions in prior 24 hours")
    dest_inflow_sum_24h: float = Field(default=0.0, ge=0.0, description="Destination inbound volume sum in prior 24 hours")


class TransactionInput(BaseModel):
    """Pydantic model representing a single live transaction submitted for fraud risk scoring.

    Enforces strict zero-leakage: any payload containing post-transaction balances
    or synthetic fraud labels is rejected at the API boundary.
    """

    request_id: Optional[str] = Field(
        default=None,
        description="Client-supplied correlation ID for end-to-end tracing and idempotent scoring",
    )
    step: int = Field(
        ...,
        ge=1,
        le=1000,
        description="Discrete simulation hour or operational step in [1, 1000]",
    )
    type: str = Field(
        ...,
        description="Transaction corridor type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER",
    )
    amount: float = Field(
        ...,
        ge=0.0,
        description="Nominal transaction amount in operational currency ($)",
    )
    nameOrig: str = Field(
        ...,
        min_length=2,
        max_length=64,
        description="Unique originating account identifier (e.g. C1234567890)",
    )
    oldbalanceOrg: float = Field(
        ...,
        ge=0.0,
        description="Sender balance strictly prior to transaction execution ($)",
    )
    nameDest: str = Field(
        ...,
        min_length=2,
        max_length=64,
        description="Unique recipient account identifier (e.g. M1234567890 or C9876543210)",
    )
    oldbalanceDest: float = Field(
        ...,
        ge=0.0,
        description="Recipient balance strictly prior to transaction execution ($)",
    )
    historical_context: Optional[HistoricalContext] = Field(
        default=None,
        description="Point-in-time velocity features from upstream feature store",
    )
    allow_cold_start: bool = Field(
        default=False,
        description="Explicitly allow cold-start fallback if historical context is missing",
    )
    include_explanation: bool = Field(
        default=False,
        description="Whether to generate an explainable TreeSHAP risk attribution card",
    )

    @model_validator(mode="before")
    @classmethod
    def check_forbidden_leakage(cls, data: Any) -> Any:
        if isinstance(data, dict):
            found_forbidden = set(data.keys()).intersection(FORBIDDEN_LEAKAGE_COLUMNS)
            if found_forbidden:
                raise ValueError(
                    f"Target leakage violation! Payload contains forbidden post-transaction fields: "
                    f"{sorted(found_forbidden)}. Rejecting inference request to prevent data leakage."
                )
        return data

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        upper_val = value.strip().upper()
        if upper_val not in VALID_TRANSACTION_TYPES:
            raise ValueError(
                f"Invalid transaction corridor '{value}'. Must be one of: {sorted(VALID_TRANSACTION_TYPES)}"
            )
        return upper_val

    def get_effective_request_id(self) -> str:
        """Returns existing request_id or generates a fresh UUID4."""
        return self.request_id or str(uuid.uuid4())


class BatchTransactionInput(BaseModel):
    """Batch scoring request payload supporting up to configured batch limits."""

    batch_id: Optional[str] = Field(
        default=None,
        description="Optional client-provided batch correlation identifier",
    )
    transactions: List[TransactionInput] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of transaction payloads to evaluate in batch (1 to 500 items)",
    )
    allow_cold_start: bool = Field(
        default=False,
        description="Allow cold-start fallback across batch items without historical context",
    )
    include_explanation: bool = Field(
        default=False,
        description="Generate SHAP explanation cards for all items in batch",
    )

    def get_effective_batch_id(self) -> str:
        return self.batch_id or str(uuid.uuid4())


"""Unit tests for Pandera schema validation, data contracts, and zero-leakage enforcement."""

import pandas as pd
import pandera as pa
import pytest
from pydantic import ValidationError

from src.ingestion.schemas import (
    FORBIDDEN_LEAKAGE_COLUMNS,
    InferenceTransactionInput,
    RawTransactionSchema,
    verify_zero_leakage,
)


class TestRawTransactionSchema:
    """Tests schema validation on raw PaySim tabular transaction batches."""

    def test_valid_dataframe_passes(self, sample_valid_raw_df: pd.DataFrame) -> None:
        """Asserts that a compliant DataFrame successfully passes Pandera validation."""
        validated_df = RawTransactionSchema.validate(sample_valid_raw_df)
        assert len(validated_df) == 10

    def test_null_value_raises_schema_error(self, sample_valid_raw_df: pd.DataFrame) -> None:
        """Asserts that introducing a null value fails validation."""
        df_invalid = sample_valid_raw_df.copy()
        df_invalid.loc[2, "amount"] = None
        with pytest.raises(pa.errors.SchemaError):
            RawTransactionSchema.validate(df_invalid)

    def test_negative_amount_raises_schema_error(self, sample_valid_raw_df: pd.DataFrame) -> None:
        """Asserts that negative transaction amounts are strictly rejected."""
        df_invalid = sample_valid_raw_df.copy()
        df_invalid.loc[0, "amount"] = -100.0
        with pytest.raises(pa.errors.SchemaError):
            RawTransactionSchema.validate(df_invalid)

    def test_negative_balance_raises_schema_error(self, sample_valid_raw_df: pd.DataFrame) -> None:
        """Asserts that negative pre-transaction balances are strictly rejected."""
        df_invalid = sample_valid_raw_df.copy()
        df_invalid.loc[1, "oldbalanceOrg"] = -50.0
        with pytest.raises(pa.errors.SchemaError):
            RawTransactionSchema.validate(df_invalid)

    def test_invalid_transaction_type_raises_schema_error(
        self, sample_valid_raw_df: pd.DataFrame
    ) -> None:
        """Asserts that unrecognized corridor types fail schema validation."""
        df_invalid = sample_valid_raw_df.copy()
        df_invalid.loc[3, "type"] = "CRYPTO_SWAP"
        with pytest.raises(pa.errors.SchemaError):
            RawTransactionSchema.validate(df_invalid)

    def test_out_of_bounds_step_raises_schema_error(
        self, sample_valid_raw_df: pd.DataFrame
    ) -> None:
        """Asserts that simulation steps outside [1, 744] fail schema validation."""
        df_invalid = sample_valid_raw_df.copy()
        df_invalid.loc[4, "step"] = 800
        with pytest.raises(pa.errors.SchemaError):
            RawTransactionSchema.validate(df_invalid)

        df_invalid.loc[4, "step"] = 0
        with pytest.raises(pa.errors.SchemaError):
            RawTransactionSchema.validate(df_invalid)


class TestZeroLeakageEnforcement:
    """Tests zero-leakage guards forbidding post-transaction balances and naive rule flags."""

    def test_verify_zero_leakage_passes_clean_feature_set(self) -> None:
        """Asserts clean feature sets pass leakage inspection."""
        clean_features = {
            "amount",
            "amount_log",
            "oldbalanceOrg",
            "oldbalanceDest",
            "balance_orig_drain_ratio",
            "orig_tx_count_24h",
        }
        # Should execute without raising
        verify_zero_leakage(clean_features)

    @pytest.mark.parametrize("forbidden_col", list(FORBIDDEN_LEAKAGE_COLUMNS))
    def test_verify_zero_leakage_catches_forbidden_columns(self, forbidden_col: str) -> None:
        """Asserts that including any forbidden column raises a ValueError."""
        tainted_features = {"amount", "oldbalanceOrg", forbidden_col}
        with pytest.raises(ValueError, match="Target leakage violation detected"):
            verify_zero_leakage(tainted_features)


class TestInferenceTransactionInput:
    """Tests Pydantic v2 validation for single real-time transaction scoring payloads."""

    def test_valid_payload_passes(self, sample_valid_transaction_dict: dict) -> None:
        """Asserts valid payload parses into strongly typed model."""
        payload = InferenceTransactionInput(**sample_valid_transaction_dict)
        assert payload.step == 12
        assert payload.type == "TRANSFER"
        assert payload.amount == 185000.50

    def test_case_insensitive_type_normalized(self, sample_valid_transaction_dict: dict) -> None:
        """Asserts lowercase corridor types are cleanly uppercased."""
        data = sample_valid_transaction_dict.copy()
        data["type"] = "cash_out"
        payload = InferenceTransactionInput(**data)
        assert payload.type == "CASH_OUT"

    def test_negative_amount_fails_validation(self, sample_valid_transaction_dict: dict) -> None:
        """Asserts negative amount raises Pydantic ValidationError."""
        data = sample_valid_transaction_dict.copy()
        data["amount"] = -500.0
        with pytest.raises(ValidationError):
            InferenceTransactionInput(**data)

    def test_invalid_type_fails_validation(self, sample_valid_transaction_dict: dict) -> None:
        """Asserts invalid corridor raises ValidationError."""
        data = sample_valid_transaction_dict.copy()
        data["type"] = "INVALID_CORRIDOR"
        with pytest.raises(ValidationError):
            InferenceTransactionInput(**data)

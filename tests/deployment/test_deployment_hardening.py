"""Unit and integration tests for Phase 6.1 deployment hardening, security, and dual-mode persistence."""

import os
import tempfile
import pandas as pd
import pytest
from src.persistence.database import DatabaseManager, mask_account_id, sanitize_db_url
from src.api.main import create_app
from dashboard.utils.formatters import mask_account


def test_sanitize_db_url_masks_passwords() -> None:
    """Verifies that database passwords in connection URLs are redacted."""
    raw_pg_url = "postgresql://fraud_admin:SuperSecretPassword123@db-host.internal:5432/fraud_db"
    sanitized = sanitize_db_url(raw_pg_url)
    assert "SuperSecretPassword123" not in sanitized
    assert sanitized == "postgresql://fraud_admin:****@db-host.internal:5432/fraud_db"

    # URL without credentials remains unchanged
    sqlite_url = "sqlite:///data/audit_scoring.db"
    assert sanitize_db_url(sqlite_url) == sqlite_url


def test_account_masking_pci_dss_aligned() -> None:
    """Verifies account masking adheres to C1****7890 masking standard."""
    assert mask_account("C1234567890") == "C1****7890"
    assert mask_account("M9876543210") == "M9****3210"
    assert mask_account("C123") == "****"
    assert mask_account(None) == "****"

    assert mask_account_id("C555544443333") == "C5****3333"
    assert mask_account_id("123") == "****"


def test_database_manager_postgres_fallback_to_sqlite() -> None:
    """Verifies that when PostgreSQL is unreachable, DatabaseManager cleanly falls back to SQLite."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fallback_db = os.path.join(tmp_dir, "fallback_test.db")
        # Provide an unreachable PostgreSQL URL on an invalid port
        bogus_pg_url = "postgresql://invalid_user:bad_pass@127.0.0.1:59999/nonexistent_db"

        # Initialize manager with unreachable PostgreSQL URL
        mgr = DatabaseManager(db_url=bogus_pg_url)
        assert not mgr.is_sqlite

        # Schema initialization should catch the connection failure and fall back
        mgr.initialize_schema()

        # Verify fallback occurred
        assert mgr.is_sqlite is True
        assert mgr.is_connected is True
        mgr.close()


def test_cors_configuration_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that CORS allowed origins can be dynamically configured via environment variables."""
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://custom-domain.com, https://dashboard.internal")
    app = create_app()

    # Find CORSMiddleware in user_middleware
    cors_middleware = None
    for mw in app.user_middleware:
        if "CORSMiddleware" in str(mw):
            cors_middleware = mw
            break

    assert cors_middleware is not None
    allowed = cors_middleware.kwargs.get("allow_origins", [])
    assert "https://custom-domain.com" in allowed
    assert "https://dashboard.internal" in allowed


def test_validation_scored_dataset_integrity() -> None:
    """Verifies that the validation partition is pre-scored for the Policy Simulator."""
    val_scored_path = "data/processed/val_evaluation_scored.parquet"
    assert os.path.exists(val_scored_path), f"Missing {val_scored_path}"

    df_val = pd.read_parquet(val_scored_path)
    assert len(df_val) == 209984
    assert df_val["step"].min() == 501
    assert df_val["step"].max() == 620
    assert "primary_score" in df_val.columns
    assert "calibrated_probability" in df_val.columns
    assert "raw_probability" in df_val.columns
    assert "anomaly_score" in df_val.columns


def test_test_partition_step_bounds_integrity() -> None:
    """Verifies that the frozen test partition begins at Step 621 (not 624)."""
    test_path = "data/processed/test.parquet"
    assert os.path.exists(test_path), f"Missing {test_path}"

    df_test = pd.read_parquet(test_path)
    assert df_test["step"].min() == 621, f"Expected min step 621, got {df_test['step'].min()}"
    assert df_test["step"].max() == 743, f"Expected max step 743, got {df_test['step'].max()}"
    assert len(df_test) == 90829


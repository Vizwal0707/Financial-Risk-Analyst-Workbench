"""Comprehensive test suite for Phase 6 Dashboard services, components, and policy simulation."""

import os
import tempfile
import sqlite3
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from dashboard.config.settings import DashboardSettings, settings
from dashboard.config.theme import COLORS, get_institutional_css
from dashboard.utils.formatters import (
    format_currency,
    format_percentage,
    format_probability,
    format_step_time,
    mask_account,
    format_latency,
)
from dashboard.services.api_client import FraudApiClient
from dashboard.services.audit_service import AuditDataService
from dashboard.services.evaluation_service import EvaluationDataService
from dashboard.services.simulation_service import PolicySimulationService


# ==============================================================================
# 1. UTILITY & FORMATTER TESTS
# ==============================================================================

def test_currency_formatting() -> None:
    """Verifies standard and abbreviated monetary formatting."""
    assert format_currency(1234.56) == "$1,234.56"
    assert format_currency(2500000.0) == "$2.50M"
    assert format_currency(3500000000.0) == "$3.50B"
    assert format_currency(0.0) == "$0.00"
    assert format_currency(None) == "$0.00"
    assert format_currency("invalid") == "$0.00"


def test_percentage_formatting() -> None:
    """Verifies percentage formatting with decimal precision."""
    assert format_percentage(0.12345, decimals=2) == "12.35%"
    assert format_percentage(0.00612, decimals=3) == "0.612%"
    assert format_percentage(1.0) == "100.00%"
    assert format_percentage(None) == "0.00%"


def test_probability_formatting() -> None:
    """Verifies scientific precision probability representation."""
    assert format_probability(0.99876) == "0.9988"
    assert format_probability(0.00012) == "0.0001"
    assert format_probability(None) == "N/A"


def test_step_time_formatting() -> None:
    """Translates PaySim simulation step into Day and Hour."""
    assert format_step_time(1) == "Step 1 (Day 1, 00:00)"
    assert format_step_time(25) == "Step 25 (Day 2, 00:00)"
    assert format_step_time(743) == "Step 743 (Day 31, 22:00)"
    assert format_step_time(None) == "Step 0 (Day 1, 00:00)"


def test_account_masking() -> None:
    """Verifies PCI-DSS account masking preserves 2-char prefix and last 4 chars."""
    assert mask_account("C1234567890") == "C1****7890"
    assert mask_account("M9876543210") == "M9****3210"
    assert mask_account("SHORT") == "SH****HORT"
    assert mask_account("C1") == "****"
    assert mask_account(None) == "****"


def test_latency_formatting() -> None:
    """Formats microseconds and milliseconds appropriately."""
    assert format_latency(0.015) == "15 μs"
    assert format_latency(14.52) == "14.52 ms"
    assert format_latency(None) == "0.0 ms"


# ==============================================================================
# 2. API CLIENT TESTS (SUCCESS & FAILURE HANDLING)
# ==============================================================================

def test_api_client_health_success() -> None:
    """Verifies API client parses successful GET /health response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"status": "healthy", "version": "1.0.0"}

    client = FraudApiClient(base_url="http://mock-server")
    with patch("httpx.Client.get", return_value=mock_resp):
        res = client.check_health()
        assert res["status"] == "healthy"
        assert res["version"] == "1.0.0"


def test_api_client_health_failure_handled_gracefully() -> None:
    """Verifies API client returns offline dictionary when connection fails."""
    client = FraudApiClient(base_url="http://nonexistent-host:9999", timeout=0.1)
    res = client.check_health()
    assert res["status"] == "offline"
    assert "error" in res


def test_api_client_scoring_success() -> None:
    """Verifies API client parses successful POST /v1/score response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"X-Cache": "MISS"}
    mock_resp.json.return_value = {
        "request_id": "test-req-01",
        "decision": "DECLINE",
        "operational_recommendation": "BLOCK_TRANSACTION",
        "calibrated_probability": 0.95,
        "risk_weighted_investigation_priority": 11.45,
    }

    client = FraudApiClient(base_url="http://mock-server")
    with patch("httpx.Client.post", return_value=mock_resp):
        payload = {"request_id": "test-req-01", "amount": 1000.0}
        res = client.score_transaction(payload)
        assert res["_success"] is True
        assert res["decision"] == "DECLINE"
        assert res["operational_recommendation"] == "BLOCK_TRANSACTION"
        assert res["_cached"] is False


def test_api_client_scoring_server_error_handled() -> None:
    """Verifies API client returns failure structure when HTTP 500 occurs."""
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    client = FraudApiClient(base_url="http://mock-server")
    with patch("httpx.Client.post", return_value=mock_resp):
        res = client.score_transaction({"amount": 100.0})
        assert res["_success"] is False
        assert res["status_code"] == 500


# ==============================================================================
# 3. AUDIT REPOSITORY SERVICE TESTS
# ==============================================================================

def test_audit_service_empty_database_handling() -> None:
    """Verifies audit service returns structured empty stats when db is empty."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "empty_audit.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE audit_scoring_logs (
                id INTEGER PRIMARY KEY,
                request_id TEXT,
                step INTEGER,
                corridor_type TEXT,
                amount REAL,
                name_orig TEXT,
                oldbalance_orig REAL,
                name_dest TEXT,
                oldbalance_dest REAL,
                raw_probability REAL,
                calibrated_probability REAL,
                risk_score REAL,
                risk_tier TEXT,
                decision TEXT,
                action_code TEXT,
                decision_reasons TEXT,
                anomaly_score REAL,
                latency_ms REAL,
                model_version TEXT,
                cold_start_applied INTEGER,
                created_at TIMESTAMP,
                explanation_json TEXT
            );
        """)
        conn.commit()
        conn.close()

        svc = AuditDataService(db_path=db_path)
        summary = svc.get_audit_summary()
        assert summary["total_count"] == 0
        assert summary["status"] == "EMPTY"

        logs = svc.get_recent_logs(limit=10)
        assert logs.empty


def test_audit_service_reads_and_masks_records() -> None:
    """Verifies audit service queries records, calculates priority score, and applies account masking."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_audit.db")
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE audit_scoring_logs (
                id INTEGER PRIMARY KEY,
                request_id TEXT,
                step INTEGER,
                corridor_type TEXT,
                amount REAL,
                name_orig TEXT,
                oldbalance_orig REAL,
                name_dest TEXT,
                oldbalance_dest REAL,
                raw_probability REAL,
                calibrated_probability REAL,
                risk_score REAL,
                risk_tier TEXT,
                decision TEXT,
                action_code TEXT,
                decision_reasons TEXT,
                anomaly_score REAL,
                latency_ms REAL,
                model_version TEXT,
                cold_start_applied INTEGER,
                created_at TIMESTAMP,
                explanation_json TEXT
            );
        """)
        conn.execute("""
            INSERT INTO audit_scoring_logs (
                request_id, step, corridor_type, amount, name_orig, oldbalance_orig,
                name_dest, oldbalance_dest, raw_probability, calibrated_probability,
                risk_score, risk_tier, decision, action_code, decision_reasons,
                anomaly_score, latency_ms, model_version, cold_start_applied
            ) VALUES (
                'test-req-123', 50, 'TRANSFER', 50000.0, 'C1234567890', 50000.0,
                'C9876543210', 0.0, 0.98, 0.92, 92.0, 'HIGH', 'DECLINE',
                'BLOCK_TRANSACTION', '["BALANCE_DRAIN_CRITICAL"]', 0.45, 1.25, '1.0.0', 1
            );
        """)
        conn.commit()
        conn.close()

        svc = AuditDataService(db_path=db_path)
        df = svc.get_recent_logs(limit=10)
        assert len(df) == 1
        assert df["name_orig_masked"].iloc[0] == "C1****7890"
        assert df["name_dest_masked"].iloc[0] == "C9****3210"
        assert df["decision"].iloc[0] == "DECLINE"
        assert df["risk_weighted_investigation_priority"].iloc[0] > 0.0

        rec = svc.get_record_by_request_id("test-req-123")
        assert rec is not None
        assert rec["request_id"] == "test-req-123"
        assert rec["name_orig_masked"] == "C1****7890"
        assert "BALANCE_DRAIN_CRITICAL" in rec["decision_reasons"]


# ==============================================================================
# 4. EVALUATION SERVICE TESTS
# ==============================================================================

def test_evaluation_service_loads_audited_reports() -> None:
    """Verifies evaluation service loads existing Phase 4/4.1 report files."""
    svc = EvaluationDataService()
    comp = svc.get_model_comparison()
    assert "metrics_by_model" in comp
    assert "XGBoost_Calibrated" in comp["metrics_by_model"]

    test_metrics = svc.get_test_metrics()
    assert "test_metrics" in test_metrics
    assert test_metrics["test_samples"] == 90829

    cm = svc.get_confusion_matrix()
    assert not cm.empty
    assert "actual_class" in cm.columns

    ablation = svc.get_feature_ablation_results()
    assert not ablation.empty
    assert len(ablation) == 11


def test_drift_monitoring_unavailable_state() -> None:
    """Verifies that drift monitoring explicitly reports an unavailable state rather than fabricating drift."""
    svc = EvaluationDataService()
    drift = svc.get_drift_status()
    assert drift["status"] == "UNAVAILABLE"
    assert "PaySim evaluation benchmark contains a frozen 743-hour simulation horizon" in drift["reason"]


# ==============================================================================
# 5. POLICY SIMULATOR SERVICE TESTS
# ==============================================================================

def test_policy_simulation_calculations() -> None:
    """Verifies vectorized policy threshold calculation on synthetic evaluation data."""
    test_df = pd.DataFrame({
        "primary_score": [0.01, 0.25, 0.85, 0.95],
        "isFraud": [0, 0, 1, 1],
        "amount": [100.0, 500.0, 10000.0, 25000.0],
    })

    sim_svc = PolicySimulationService()
    res = sim_svc.simulate(
        tau_low=0.10,
        tau_high=0.75,
        c_fp=100.0,
        c_rev=25.0,
        daily_capacity=500,
        df=test_df,
    )

    vol = res["volumes"]
    assert vol["approve"] == 1  # 0.01 < 0.10
    assert vol["review"] == 1   # 0.25 in [0.10, 0.75)
    assert vol["decline"] == 2  # 0.85 and 0.95 >= 0.75

    cm = res["confusion_matrix"]
    assert cm["legit_approve"] == 1
    assert cm["legit_review"] == 1
    assert cm["legit_decline"] == 0
    assert cm["fraud_approve"] == 0
    assert cm["fraud_review"] == 0
    assert cm["fraud_decline"] == 2

    # Financial impact
    fin = res["financial_impact"]
    assert fin["investigation_costs"] == 25.0  # 1 review * $25
    assert fin["false_positive_costs"] == 0.0  # 0 legit in decline
    assert res["amounts"]["captured_fraud_value"] == 35000.0  # 10000 + 25000
    assert fin["net_savings"] == 35000.0 - 25.0


def test_policy_simulation_invalid_thresholds_handling() -> None:
    """Verifies policy simulator handles edge threshold boundaries."""
    test_df = pd.DataFrame({
        "primary_score": [0.05, 0.50, 0.90],
        "isFraud": [0, 0, 1],
        "amount": [50.0, 200.0, 5000.0],
    })
    sim_svc = PolicySimulationService()
    
    # Boundary: tau_low = tau_high (0-width review band)
    res = sim_svc.simulate(tau_low=0.50, tau_high=0.50, df=test_df)
    assert res["volumes"]["review"] == 0
    assert res["volumes"]["approve"] == 1
    assert res["volumes"]["decline"] == 2


"""Comprehensive regression, security, durability, and action-separation tests for Phase 5.1 Audit."""

import asyncio
import os
import sqlite3
import tempfile
import uuid
import pytest
import httpx
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.api.main import app
from src.persistence.database import DatabaseManager, mask_account_id
from src.decision_engine.policy_engine import DecisionPolicyEngine


@pytest.fixture(scope="module")
def client():
    """Module-scoped FastAPI TestClient running application lifespan."""
    with TestClient(app) as test_client:
        yield test_client


# ==============================================================================
# 1. RISK DECISION VS OPERATIONAL RECOMMENDATION SEPARATION
# ==============================================================================

def test_operational_recommendation_and_disclaimer_fields(client: TestClient) -> None:
    """Verifies that scoring responses include separated operational recommendation and safety disclaimers."""
    payload = {
        "request_id": f"rec-sep-{uuid.uuid4()}",
        "step": 50,
        "type": "PAYMENT",
        "amount": 25.0,
        "nameOrig": "C199999901",
        "oldbalanceOrg": 500.0,
        "nameDest": "M299999901",
        "oldbalanceDest": 0.0,
        "allow_cold_start": True,
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "risk_decision" in data
    assert "operational_recommendation" in data
    assert "simulated_recommendation_only" in data
    assert data["simulated_recommendation_only"] is True
    assert "recommendation_disclaimer" in data
    assert "prohibited from triggering real payment authorization" in data["recommendation_disclaimer"]

    # Verify risk decision and operational recommendation mapping
    assert data["risk_decision"] == "APPROVE"
    assert data["operational_recommendation"] == "STP_APPROVE"
    # Backwards-compatible aliases
    assert data["action_code"] == "STP_APPROVE"
    assert data["decision"] == "APPROVE"


def test_review_recommendation_mapping() -> None:
    """Verifies REVIEW risk decision maps to QUEUE_L1_REVIEW."""
    policy = DecisionPolicyEngine(tau_low=0.10, tau_high=0.75)
    result = policy.evaluate_decision(
        calibrated_prob=0.45,
        amount=10000.0,
        features_dict={"is_transfer": 1.0, "is_cash_out": 0.0, "orig_balance_drain": 0.5},
        anomaly_score=0.3,
        cold_start_applied=True,
    )
    assert result.decision == "REVIEW"
    assert result.operational_recommendation == "QUEUE_L1_REVIEW"


def test_decline_block_transaction_vs_step_up_mapping() -> None:
    """Verifies DECLINE with severe risk triggers BLOCK_TRANSACTION while moderate triggers REQUIRE_STEP_UP_AUTHENTICATION."""
    policy = DecisionPolicyEngine(tau_low=0.10, tau_high=0.75)

    # Extreme risk / critical drain -> BLOCK_TRANSACTION
    res_block = policy.evaluate_decision(
        calibrated_prob=0.95,
        amount=250000.0,
        features_dict={
            "is_transfer": 1.0,
            "is_cash_out": 0.0,
            "orig_balance_drain": 1.0,
            "is_orig_zero_balance_post": 1.0,
        },
        anomaly_score=0.99,
        cold_start_applied=True,
    )
    assert res_block.decision == "DECLINE"
    assert res_block.operational_recommendation == "BLOCK_TRANSACTION"

    # Moderate decline prob without liquidation -> REQUIRE_STEP_UP_AUTHENTICATION
    res_step_up = policy.evaluate_decision(
        calibrated_prob=0.80,
        amount=1500.0,
        features_dict={
            "is_transfer": 1.0,
            "is_cash_out": 0.0,
            "orig_balance_drain": 0.2,
            "is_orig_zero_balance_post": 0.0,
        },
        anomaly_score=0.5,
        cold_start_applied=True,
    )
    assert res_step_up.decision == "DECLINE"
    assert res_step_up.operational_recommendation == "REQUIRE_STEP_UP_AUTHENTICATION"


# ==============================================================================
# 2. RISK-WEIGHTED INVESTIGATION PRIORITY & TRIAGE SCORE
# ==============================================================================

def test_risk_weighted_investigation_priority_semantics(client: TestClient) -> None:
    """Verifies that risk_weighted_investigation_priority computes P_cal * ln(1 + amount)."""
    import math

    amount = 54321.0
    payload = {
        "request_id": f"prio-{uuid.uuid4()}",
        "step": 100,
        "type": "CASH_OUT",
        "amount": amount,
        "nameOrig": "C123456789",
        "oldbalanceOrg": 54321.0,
        "nameDest": "C987654321",
        "oldbalanceDest": 0.0,
        "allow_cold_start": True,
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    p_cal = data["calibrated_fraud_probability"]
    expected_priority = round(float(p_cal * math.log1p(amount)), 4)

    assert data["risk_weighted_investigation_priority"] == pytest.approx(expected_priority, abs=0.01)
    # Check backwards compatible alias triage_score
    assert data["triage_score"] == pytest.approx(expected_priority, abs=0.01)


# ==============================================================================
# 3. REASON CODE EVIDENCE GROUNDING & SUPPRESSION
# ==============================================================================

def test_merchant_payment_never_triggers_zero_balance_mule() -> None:
    """Authentic merchant payees ('M...') must never trigger DEST_ZERO_BALANCE_MULE even if destination balance is 0.0."""
    policy = DecisionPolicyEngine(tau_low=0.10, tau_high=0.75)
    feats = {
        "is_transfer": 0.0,
        "is_cash_out": 0.0,
        "is_payment": 1.0,
        "is_merchant": 1.0,  # Merchant payee
        "oldbalanceDest": 0.0,
        "orig_balance_drain": 0.05,
    }
    result = policy.evaluate_decision(
        calibrated_prob=0.02,
        amount=50.0,
        features_dict=feats,
        anomaly_score=None,
        cold_start_applied=True,
    )
    assert "DEST_ZERO_BALANCE_MULE" not in result.reason_codes


def test_cold_start_suppresses_velocity_reason_codes() -> None:
    """When cold_start_applied is True, velocity reason codes must be suppressed due to absence of historical record."""
    policy = DecisionPolicyEngine(tau_low=0.10, tau_high=0.75)
    feats = {
        "is_transfer": 1.0,
        "is_cash_out": 0.0,
        "velocity_tx_count_1h": 25.0,  # Would trigger if not suppressed
        "velocity_tx_count_24h": 100.0,
        "orig_balance_drain": 0.2,
    }
    result = policy.evaluate_decision(
        calibrated_prob=0.30,
        amount=500.0,
        features_dict=feats,
        anomaly_score=None,
        cold_start_applied=True,  # Cold start applied!
    )
    assert "ELEVATED_VELOCITY_1H" not in result.reason_codes
    assert "ELEVATED_VELOCITY_24H" not in result.reason_codes


# ==============================================================================
# 4. PII ACCOUNT MASKING AUDIT
# ==============================================================================

def test_pii_account_masking() -> None:
    """Account IDs must be masked with only prefix and last 4 characters visible."""
    assert mask_account_id("C1234567890") == "C1****7890"
    assert mask_account_id("M9876543210") == "M9****3210"
    assert mask_account_id("SHORT") == "SH****HORT"
    assert mask_account_id("C1") == "****"
    assert mask_account_id(None) == "****"


# ==============================================================================
# 5. DATABASE DURABILITY, RESILIENCE & FAILURE RECOVERY
# ==============================================================================

def test_scoring_resilience_to_database_failure(client: TestClient) -> None:
    """If database write fails, the scoring API MUST still return HTTP 200 with valid fraud score without throwing 500."""
    audit_repo = app.state.audit_repo

    with patch.object(audit_repo, "log_decision", side_effect=sqlite3.OperationalError("disk I/O error")):
        payload = {
            "request_id": f"resilience-{uuid.uuid4()}",
            "step": 200,
            "type": "CASH_OUT",
            "amount": 1000.0,
            "nameOrig": "C555555555",
            "oldbalanceOrg": 1000.0,
            "nameDest": "C666666666",
            "oldbalanceDest": 500.0,
            "allow_cold_start": True,
        }
        resp = client.post("/v1/score", json=payload)
        # Even with DB down, client scoring SLA is preserved
        assert resp.status_code == 200
        data = resp.json()
        assert "risk_decision" in data
        assert "calibrated_fraud_probability" in data


def test_database_idempotency_across_restarts() -> None:
    """A scored transaction persisted to disk SQLite database must retain idempotency across engine restarts."""
    from src.persistence.audit_repository import AuditRepository

    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "restart_test.db")
        db_mgr_1 = DatabaseManager(f"sqlite:///{db_path}")
        db_mgr_1.initialize_schema()
        repo_1 = AuditRepository(db_mgr_1)

        req_id = f"restart-idemp-{uuid.uuid4()}"

        async def _persist():
            await repo_1.log_decision(
                request_id=req_id,
                step=300,
                corridor_type="TRANSFER",
                amount=9999.0,
                name_orig="C111111111",
                oldbalance_orig=9999.0,
                name_dest="C222222222",
                oldbalance_dest=0.0,
                raw_probability=0.88,
                calibrated_probability=0.92,
                risk_score=92.0,
                risk_tier="HIGH",
                decision="DECLINE",
                action_code="BLOCK_TRANSACTION",
                decision_reasons=["BALANCE_DRAIN_CRITICAL"],
                latency_ms=1.5,
                model_version="1.0.0",
                anomaly_score=None,
                cold_start_applied=True,
            )

        asyncio.run(_persist())
        db_mgr_1.close()

        # Restart 2: Fresh DatabaseManager opening the same physical file
        db_mgr_2 = DatabaseManager(f"sqlite:///{db_path}")
        repo_2 = AuditRepository(db_mgr_2)

        async def _retrieve():
            return await repo_2.get_by_request_id(req_id)

        existing = asyncio.run(_retrieve())
        assert existing is not None
        assert existing["request_id"] == req_id
        assert existing["decision"] == "DECLINE"
        assert existing["operational_recommendation"] == "BLOCK_TRANSACTION"
        assert existing["calibrated_probability"] == pytest.approx(0.92)
        db_mgr_2.close()


# ==============================================================================
# 6. CONCURRENT ASGI REQUEST STABILITY
# ==============================================================================

def test_concurrent_asgi_requests() -> None:
    """Validates that 10 concurrent async HTTP requests succeed with 0% error rate."""
    async def _runner() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=15.0) as client:
            tasks = []
            for i in range(10):
                payload = {
                    "request_id": f"async-conc-{i}-{uuid.uuid4()}",
                    "step": 150,
                    "type": "TRANSFER",
                    "amount": float(100 + i * 50),
                    "nameOrig": f"C7777777{i:02d}",
                    "oldbalanceOrg": float(100 + i * 50),
                    "nameDest": f"C8888888{i:02d}",
                    "oldbalanceDest": 0.0,
                    "allow_cold_start": True,
                }
                tasks.append(client.post("/v1/score", json=payload))

            responses = await asyncio.gather(*tasks)
            for resp in responses:
                assert resp.status_code == 200
                data = resp.json()
                assert data["risk_decision"] in ("APPROVE", "REVIEW", "DECLINE")
                assert "risk_weighted_investigation_priority" in data

    asyncio.run(_runner())

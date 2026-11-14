"""Unit and functional tests for FastAPI endpoint routes, health, governance, and scoring."""

import uuid
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    """Module-scoped FastAPI TestClient running the full application lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def test_liveness_health_endpoint(client: TestClient) -> None:
    """GET /health must return HTTP 200 and healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["version"] == "1.0.0"


def test_readiness_probe_endpoint(client: TestClient) -> None:
    """GET /ready must return HTTP 200, verifying models loaded and DB connected."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["models_loaded"] is True
    assert data["database_connected"] is True
    assert data["details"]["xgboost_loaded"] is True
    assert data["details"]["calibrator_loaded"] is True


def test_model_info_governance_endpoint(client: TestClient) -> None:
    """GET /v1/model/info returns model lineage, 31 features, and audited thresholds."""
    resp = client.get("/v1/model/info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_name"] == "PaySim_CostWeightedXGBoost_Calibrated"
    assert data["n_features"] == 31
    assert len(data["feature_names"]) == 31
    assert data["calibration_method"] == "isotonic"
    assert "decision_policy" in data
    assert data["decision_policy"]["tau_low"] == 0.10
    assert data["decision_policy"]["tau_high"] == 0.75
    assert "cost_assumptions" in data
    assert "audit_disclaimer" in data


def test_score_single_legitimate_approval(client: TestClient) -> None:
    """Low risk small merchant payment should evaluate to APPROVE (Straight-Through Processing)."""
    payload = {
        "request_id": "test-route-approve-01",
        "step": 20,
        "type": "PAYMENT",
        "amount": 42.50,
        "nameOrig": "C1000000001",
        "oldbalanceOrg": 5000.0,
        "nameDest": "M9000000001",
        "oldbalanceDest": 0.0,
        "allow_cold_start": True,
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] == "APPROVE"
    assert data["action_code"] == "STP_APPROVE"
    assert data["risk_tier"] == "LOW"
    assert data["calibrated_probability"] < 0.10
    assert data["cold_start_applied"] is True
    assert "LOW_RISK_MERCHANT_PAYMENT" in data["decision_reasons"] or "STP_NORMAL_ACTIVITY" in data["decision_reasons"]


def test_score_single_fraud_decline_with_explanation(client: TestClient) -> None:
    """High risk total account liquidation in TRANSFER corridor must DECLINE with SHAP card."""
    payload = {
        "request_id": "test-route-decline-01",
        "step": 45,
        "type": "TRANSFER",
        "amount": 350000.0,
        "nameOrig": "C2000000002",
        "oldbalanceOrg": 350000.0,  # 100% balance drain
        "nameDest": "C3000000003",
        "oldbalanceDest": 0.0,       # Synthetic mule destination
        "allow_cold_start": True,
        "include_explanation": True,
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] == "DECLINE"
    assert data["action_code"] == "AUTO_DECLINE_STEP_UP"
    assert data["risk_tier"] == "HIGH"
    assert data["calibrated_probability"] >= 0.75
    assert "BALANCE_DRAIN_CRITICAL" in data["decision_reasons"]
    assert "CORRIDOR_TRANSFER_HIGH_RISK" in data["decision_reasons"]

    # Verify SHAP explanation card presence
    assert data["explanation"] is not None
    expl = data["explanation"]
    assert len(expl["top_risk_factors"]) > 0
    assert "narrative" in expl


def test_score_idempotency_and_cache_header(client: TestClient) -> None:
    """Submitting the exact same request_id must return cached decision with X-Cache: HIT."""
    unique_id = f"test-route-idempotent-{uuid.uuid4()}"
    payload = {
        "request_id": unique_id,
        "step": 12,
        "type": "CASH_OUT",
        "amount": 15000.0,
        "nameOrig": "C4000000004",
        "oldbalanceOrg": 20000.0,
        "nameDest": "C5000000005",
        "oldbalanceDest": 1000.0,
        "allow_cold_start": True,
    }
    resp1 = client.post("/v1/score", json=payload)
    assert resp1.status_code == 200
    assert resp1.headers.get("x-cache") == "MISS"
    orig_score = resp1.json()["calibrated_probability"]

    # Second invocation with same request_id
    resp2 = client.post("/v1/score", json=payload)
    assert resp2.status_code == 200
    assert resp2.headers.get("x-cache") == "HIT"
    assert resp2.json()["calibrated_probability"] == orig_score
    assert resp2.json()["request_id"] == unique_id


def test_score_batch_success(client: TestClient) -> None:
    """POST /v1/score/batch scores multiple transactions and aggregates summary metrics."""
    batch_payload = {
        "batch_id": "test-batch-001",
        "allow_cold_start": True,
        "transactions": [
            {
                "request_id": "batch-item-01",
                "step": 10,
                "type": "PAYMENT",
                "amount": 15.0,
                "nameOrig": "C6000000006",
                "oldbalanceOrg": 1000.0,
                "nameDest": "M1111111111",
                "oldbalanceDest": 0.0,
            },
            {
                "request_id": "batch-item-02",
                "step": 10,
                "type": "TRANSFER",
                "amount": 500000.0,
                "nameOrig": "C7000000007",
                "oldbalanceOrg": 500000.0,
                "nameDest": "C8000000008",
                "oldbalanceDest": 0.0,
            },
        ],
    }
    resp = client.post("/v1/score/batch", json=batch_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["batch_id"] == "test-batch-001"
    summary = data["summary"]
    assert summary["total_evaluated"] == 2
    assert summary["approvals"] + summary["reviews"] + summary["declines"] == 2
    assert len(data["results"]) == 2
    assert data["total_latency_ms"] > 0.0


def test_score_batch_empty_rejected(client: TestClient) -> None:
    """Batch with 0 transactions must be rejected with HTTP 422."""
    resp = client.post("/v1/score/batch", json={"transactions": []})
    assert resp.status_code == 422

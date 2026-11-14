"""End-to-end integration tests for serving pipeline, numerical parity, and audit persistence."""

import uuid
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_serving_pipeline_end_to_end(client: TestClient) -> None:
    """Verifies complete inference lifecycle: feature creation -> XGBoost -> Isotonic -> Decision -> DB."""
    req_id = "test-integ-e2e-001"
    payload = {
        "request_id": req_id,
        "step": 33,
        "type": "TRANSFER",
        "amount": 180000.0,
        "nameOrig": "C555111222",
        "oldbalanceOrg": 180000.0,
        "nameDest": "C999888777",
        "oldbalanceDest": 0.0,
        "allow_cold_start": True,
        "include_explanation": True,
    }

    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["request_id"] == req_id
    assert data["decision"] in ("APPROVE", "REVIEW", "DECLINE")
    assert 0.0 <= data["calibrated_probability"] <= 1.0
    assert 0.0 <= data["risk_score"] <= 100.0
    assert len(data["decision_reasons"]) > 0
    assert data["latency_ms"] > 0.0
    assert data["explanation"] is not None
    assert len(data["explanation"]["top_risk_factors"]) > 0

    # Query audit database to verify persistence
    db_mgr = app.state.db_manager
    rows = client.app.state.db_manager.get_connection().execute(
        "SELECT * FROM audit_scoring_logs WHERE request_id = ?", (req_id,)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["request_id"] == req_id
    assert rows[0]["decision"] == data["decision"]


def test_batch_vs_single_numerical_exact_parity(client: TestClient) -> None:
    """Rigorous verification: individual score outputs must match batch outputs with 100% numerical parity."""
    # Create 20 synthetic transactions spanning diverse corridors and amounts
    test_cases = [
        {"step": 1, "type": "PAYMENT", "amount": 10.50, "nameOrig": "C101", "oldbalanceOrg": 200.0, "nameDest": "M201", "oldbalanceDest": 0.0},
        {"step": 5, "type": "TRANSFER", "amount": 250000.0, "nameOrig": "C102", "oldbalanceOrg": 250000.0, "nameDest": "C202", "oldbalanceDest": 0.0},
        {"step": 12, "type": "CASH_OUT", "amount": 80000.0, "nameOrig": "C103", "oldbalanceOrg": 100000.0, "nameDest": "C203", "oldbalanceDest": 50000.0},
        {"step": 20, "type": "CASH_IN", "amount": 5000.0, "nameOrig": "C104", "oldbalanceOrg": 1000.0, "nameDest": "C204", "oldbalanceDest": 10000.0},
        {"step": 25, "type": "DEBIT", "amount": 350.0, "nameOrig": "C105", "oldbalanceOrg": 4000.0, "nameDest": "C205", "oldbalanceDest": 2000.0},
        {"step": 30, "type": "TRANSFER", "amount": 1000000.0, "nameOrig": "C106", "oldbalanceOrg": 1000000.0, "nameDest": "C206", "oldbalanceDest": 0.0},
        {"step": 45, "type": "PAYMENT", "amount": 89.99, "nameOrig": "C107", "oldbalanceOrg": 850.0, "nameDest": "M207", "oldbalanceDest": 0.0},
        {"step": 50, "type": "CASH_OUT", "amount": 15000.0, "nameOrig": "C108", "oldbalanceOrg": 15000.0, "nameDest": "C208", "oldbalanceDest": 0.0},
    ]

    single_results = []
    batch_tx_list = []

    run_uuid = uuid.uuid4()
    for idx, tc in enumerate(test_cases):
        req_id = f"parity-single-{run_uuid}-{idx:03d}"
        payload = {
            "request_id": req_id,
            "step": tc["step"],
            "type": tc["type"],
            "amount": tc["amount"],
            "nameOrig": tc["nameOrig"],
            "oldbalanceOrg": tc["oldbalanceOrg"],
            "nameDest": tc["nameDest"],
            "oldbalanceDest": tc["oldbalanceDest"],
            "allow_cold_start": True,
        }
        # Score single
        s_resp = client.post("/v1/score", json=payload)
        assert s_resp.status_code == 200
        single_results.append(s_resp.json())

        # Prepare for batch with unique request_id
        batch_payload = dict(payload)
        batch_payload["request_id"] = f"parity-batch-{run_uuid}-{idx:03d}"
        batch_tx_list.append(batch_payload)

    # Score in batch
    b_resp = client.post(
        "/v1/score/batch",
        json={"batch_id": f"parity-batch-run-{run_uuid}", "transactions": batch_tx_list, "allow_cold_start": True},
    )
    assert b_resp.status_code == 200
    batch_results = b_resp.json()["results"]

    assert len(single_results) == len(batch_results)

    for i in range(len(single_results)):
        s = single_results[i]
        b = batch_results[i]

        prob_diff = abs(s["calibrated_probability"] - b["calibrated_probability"])
        raw_prob_diff = abs(s["raw_probability"] - b["raw_probability"])
        score_diff = abs(s["risk_score"] - b["risk_score"])

        assert prob_diff < 1e-5, f"Item {i}: Calibrated probability mismatch {s['calibrated_probability']} vs {b['calibrated_probability']}"
        assert raw_prob_diff < 1e-5, f"Item {i}: Raw probability mismatch {s['raw_probability']} vs {b['raw_probability']}"
        assert score_diff < 1e-2, f"Item {i}: Risk score mismatch {s['risk_score']} vs {b['risk_score']}"
        assert s["decision"] == b["decision"], f"Item {i}: Decision mismatch {s['decision']} vs {b['decision']}"
        assert s["action_code"] == b["action_code"], f"Item {i}: Action code mismatch {s['action_code']} vs {b['action_code']}"
        assert s["risk_tier"] == b["risk_tier"], f"Item {i}: Risk tier mismatch {s['risk_tier']} vs {b['risk_tier']}"


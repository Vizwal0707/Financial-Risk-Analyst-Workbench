"""API boundary input validation and anti-leakage compliance tests."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_negative_amount_fails_validation(client: TestClient) -> None:
    """Negative transaction amount must be rejected at API schema boundary."""
    payload = {
        "step": 1,
        "type": "PAYMENT",
        "amount": -100.0,
        "nameOrig": "C12345",
        "oldbalanceOrg": 500.0,
        "nameDest": "M54321",
        "oldbalanceDest": 0.0,
        "allow_cold_start": True,
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 422
    assert "amount" in resp.text


def test_negative_balance_fails_validation(client: TestClient) -> None:
    """Negative pre-transaction balance must be rejected."""
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 100.0,
        "nameOrig": "C12345",
        "oldbalanceOrg": -50.0,
        "nameDest": "C54321",
        "oldbalanceDest": 0.0,
        "allow_cold_start": True,
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 422
    assert "oldbalanceOrg" in resp.text


def test_invalid_corridor_type_fails_validation(client: TestClient) -> None:
    """Non-PaySim corridor type must be rejected."""
    payload = {
        "step": 1,
        "type": "WIRE_TRANSFER",
        "amount": 100.0,
        "nameOrig": "C12345",
        "oldbalanceOrg": 500.0,
        "nameDest": "C54321",
        "oldbalanceDest": 0.0,
        "allow_cold_start": True,
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 422
    assert "WIRE_TRANSFER" in resp.text or "type" in resp.text


@pytest.mark.parametrize(
    "forbidden_field,forbidden_value",
    [
        ("isFraud", 1),
        ("is_fraud", 1),
        ("newbalanceOrig", 0.0),
        ("newbalance_orig", 0.0),
        ("newbalanceDest", 500.0),
        ("newbalance_dest", 500.0),
        ("isFlaggedFraud", 1),
        ("is_flagged_fraud", 1),
    ],
)
def test_forbidden_leakage_fields_rejected_at_boundary(
    client: TestClient, forbidden_field: str, forbidden_value: object
) -> None:
    """Strict anti-leakage guard: presence of any forbidden post-transaction field must trigger HTTP 422."""
    payload = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 500.0,
        "nameOrig": "C12345",
        "oldbalanceOrg": 1000.0,
        "nameDest": "C54321",
        "oldbalanceDest": 0.0,
        "allow_cold_start": True,
        forbidden_field: forbidden_value,
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert data["type"] == "urn:problem:data-leakage-violation" or "leakage" in data["detail"].lower()


def test_missing_historical_context_without_cold_start_rejected(client: TestClient) -> None:
    """When allow_cold_start is False and historical context is missing, must reject with structured 422."""
    payload = {
        "step": 5,
        "type": "TRANSFER",
        "amount": 1000.0,
        "nameOrig": "C99999",
        "oldbalanceOrg": 1000.0,
        "nameDest": "C88888",
        "oldbalanceDest": 0.0,
        "allow_cold_start": False,  # Strict mode requires historical context
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 422
    data = resp.json()
    assert data["type"] == "urn:problem:missing-historical-context"
    assert "historical feature context missing" in data["detail"].lower()


def test_provided_historical_context_consumed_cleanly(client: TestClient) -> None:
    """When valid historical context is supplied, scoring succeeds with cold_start_applied=False."""
    payload = {
        "step": 10,
        "type": "TRANSFER",
        "amount": 2000.0,
        "nameOrig": "C99999",
        "oldbalanceOrg": 5000.0,
        "nameDest": "C88888",
        "oldbalanceDest": 100.0,
        "allow_cold_start": False,
        "historical_context": {
            "orig_tx_count_1h": 1,
            "orig_tx_count_6h": 2,
            "orig_tx_count_24h": 3,
            "orig_tx_count_72h": 4,
            "orig_amount_sum_24h": 4500.0,
            "orig_amount_mean_24h": 1500.0,
            "orig_amount_std_24h": 350.0,
            "dest_inflow_count_24h": 1,
            "dest_inflow_sum_24h": 1000.0,
        },
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["cold_start_applied"] is False
    assert data["calibrated_probability"] >= 0.0


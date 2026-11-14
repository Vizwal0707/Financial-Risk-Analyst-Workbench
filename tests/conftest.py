"""Pytest test configuration and shared fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_valid_transaction_dict() -> dict:
    """Returns a valid individual transaction dictionary for payload tests."""
    return {
        "step": 12,
        "type": "TRANSFER",
        "amount": 185000.50,
        "nameOrig": "C1234567890",
        "oldbalanceOrg": 250000.00,
        "nameDest": "C9876543210",
        "oldbalanceDest": 10000.00,
    }


@pytest.fixture
def sample_valid_raw_df() -> pd.DataFrame:
    """Returns a synthetic 10-row DataFrame strictly conforming to RawTransactionSchema."""
    return pd.DataFrame(
        {
            "step": [1, 5, 10, 24, 100, 250, 400, 505, 625, 740],
            "type": [
                "PAYMENT",
                "CASH_OUT",
                "TRANSFER",
                "CASH_IN",
                "DEBIT",
                "PAYMENT",
                "TRANSFER",
                "CASH_OUT",
                "TRANSFER",
                "CASH_OUT",
            ],
            "amount": [
                150.0,
                4500.0,
                98000.0,
                12000.0,
                50.0,
                320.0,
                450000.0,
                1200.0,
                890000.0,
                50000.0,
            ],
            "nameOrig": [f"C100000000{i}" for i in range(10)],
            "oldbalanceOrg": [
                1000.0,
                5000.0,
                100000.0,
                0.0,
                500.0,
                1200.0,
                450000.0,
                2000.0,
                890000.0,
                60000.0,
            ],
            "newbalanceOrig": [
                850.0,
                500.0,
                2000.0,
                12000.0,
                450.0,
                880.0,
                0.0,
                800.0,
                0.0,
                10000.0,
            ],
            "nameDest": [f"M200000000{i}" if i % 2 == 0 else f"C300000000{i}" for i in range(10)],
            "oldbalanceDest": [0.0, 2000.0, 5000.0, 50000.0, 0.0, 0.0, 0.0, 10000.0, 0.0, 25000.0],
            "newbalanceDest": [
                0.0,
                6500.0,
                103000.0,
                38000.0,
                0.0,
                0.0,
                450000.0,
                11200.0,
                890000.0,
                75000.0,
            ],
            "isFraud": [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            "isFlaggedFraud": [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        }
    )

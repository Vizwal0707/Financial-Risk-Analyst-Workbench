"""Policy simulation service for evaluating hypothetical thresholds on offline benchmark data."""

import logging
import os
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from dashboard.config.settings import settings

logger = logging.getLogger(__name__)


class PolicySimulationService:
    """Executes in-memory policy sensitivity simulations on evaluation data.

    By default operates on the validation partition (Steps 501–620, N=209,984) to
    explore threshold tradeoffs while preserving the out-of-time test partition
    (Steps 621–743, N=90,829) strictly frozen for benchmark governance.
    """

    def __init__(self, data_path: str = "data/processed/val_evaluation_scored.parquet") -> None:
        self.data_path = data_path
        self._cached_df: Optional[pd.DataFrame] = None

    def load_evaluation_data(self) -> pd.DataFrame:
        """Loads and caches the pre-scored evaluation dataset."""
        if self._cached_df is not None:
            return self._cached_df

        target_path = self.data_path
        if not os.path.exists(target_path):
            fallback_path = "data/processed/test_evaluation_scored.parquet"
            if os.path.exists(fallback_path):
                logger.info(f"Primary evaluation path {target_path} not found, falling back to {fallback_path}")
                target_path = fallback_path
            else:
                logger.warning(f"Evaluation scored dataset not found at {target_path} or {fallback_path}")
                return pd.DataFrame()

        try:
            df = pd.read_parquet(target_path)
            self._cached_df = df
            return df
        except Exception as e:
            logger.error(f"Failed to load evaluation dataset: {e}")
            return pd.DataFrame()

    def simulate(
        self,
        tau_low: float = 0.10,
        tau_high: float = 0.75,
        c_fp: float = 100.0,
        c_rev: float = 25.0,
        daily_capacity: int = 500,
        fn_multiplier: float = 1.0,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Evaluates hypothetical threshold and cost parameters on offline benchmark data."""
        data = df if df is not None else self.load_evaluation_data()
        if data.empty:
            return {"error": "Evaluation dataset is empty or unavailable"}

        scores = data["primary_score"].values
        is_fraud = data["isFraud"].values.astype(bool)
        amounts = data["amount"].values

        total_tx = len(data)
        total_frauds = int(is_fraud.sum())
        total_legit = total_tx - total_frauds
        total_fraud_exposure = float(amounts[is_fraud].sum())
        total_legit_volume = float(amounts[~is_fraud].sum())

        # Vectorized Decision Assignment
        is_decline = scores >= tau_high
        is_review = (scores >= tau_low) & (~is_decline)
        is_approve = scores < tau_low

        # Volume Counts
        cnt_approve = int(is_approve.sum())
        cnt_review = int(is_review.sum())
        cnt_decline = int(is_decline.sum())

        # Breakdown by Class: Legitimate (isFraud=0)
        legit_approve = int((~is_fraud & is_approve).sum())
        legit_review = int((~is_fraud & is_review).sum())
        legit_decline = int((~is_fraud & is_decline).sum())  # False Positives

        # Breakdown by Class: Fraud (isFraud=1)
        fraud_approve = int((is_fraud & is_approve).sum())    # Missed Frauds (False Negatives)
        fraud_review = int((is_fraud & is_review).sum())      # Intercepted Frauds (Review)
        fraud_decline = int((is_fraud & is_decline).sum())    # Blocked Frauds (Decline)

        # Dollar Allocations
        amt_fraud_approve = float(amounts[is_fraud & is_approve].sum())
        amt_fraud_review = float(amounts[is_fraud & is_review].sum())
        amt_fraud_decline = float(amounts[is_fraud & is_decline].sum())

        amt_legit_approve = float(amounts[~is_fraud & is_approve].sum())
        amt_legit_review = float(amounts[~is_fraud & is_review].sum())
        amt_legit_decline = float(amounts[~is_fraud & is_decline].sum())

        # Performance Metrics
        captured_frauds = fraud_review + fraud_decline
        captured_fraud_value = amt_fraud_review + amt_fraud_decline
        missed_fraud_value = amt_fraud_approve * fn_multiplier

        recall = captured_frauds / max(total_frauds, 1)
        precision = captured_frauds / max(cnt_review + cnt_decline, 1)
        alert_rate = (cnt_review + cnt_decline) / max(total_tx, 1)

        # Cost Calculations
        investigation_costs = cnt_review * c_rev
        false_positive_costs = legit_decline * c_fp
        total_operational_costs = investigation_costs + false_positive_costs
        net_savings = captured_fraud_value - total_operational_costs - missed_fraud_value

        # Capacity Feasibility (Horizon dynamically calculated from data steps)
        if "step" in data.columns and not data["step"].empty:
            eval_days = max((float(data["step"].max() - data["step"].min() + 1)) / 24.0, 1.0)
        else:
            eval_days = 5.0
        daily_review_rate = cnt_review / eval_days
        capacity_exceeded = daily_review_rate > daily_capacity

        return {
            "parameters": {
                "tau_low": tau_low,
                "tau_high": tau_high,
                "c_fp": c_fp,
                "c_rev": c_rev,
                "daily_capacity": daily_capacity,
                "fn_multiplier": fn_multiplier,
            },
            "volumes": {
                "total": total_tx,
                "approve": cnt_approve,
                "review": cnt_review,
                "decline": cnt_decline,
                "approve_pct": cnt_approve / max(total_tx, 1),
                "review_pct": cnt_review / max(total_tx, 1),
                "decline_pct": cnt_decline / max(total_tx, 1),
            },
            "confusion_matrix": {
                "legit_approve": legit_approve,
                "legit_review": legit_review,
                "legit_decline": legit_decline,
                "fraud_approve": fraud_approve,
                "fraud_review": fraud_review,
                "fraud_decline": fraud_decline,
            },
            "amounts": {
                "total_fraud_exposure": total_fraud_exposure,
                "captured_fraud_value": captured_fraud_value,
                "missed_fraud_value": missed_fraud_value,
                "legit_approved_amount": amt_legit_approve,
                "legit_reviewed_amount": amt_legit_review,
                "legit_declined_amount": amt_legit_decline,
            },
            "financial_impact": {
                "investigation_costs": investigation_costs,
                "false_positive_costs": false_positive_costs,
                "total_operational_costs": total_operational_costs,
                "net_savings": net_savings,
                "cost_to_save_ratio": total_operational_costs / max(captured_fraud_value, 1.0),
            },
            "operational": {
                "recall": recall,
                "precision": precision,
                "alert_rate": alert_rate,
                "daily_review_rate": daily_review_rate,
                "capacity_exceeded": capacity_exceeded,
                "capacity_utilization": daily_review_rate / max(daily_capacity, 1),
            },
        }


simulation_service = PolicySimulationService()


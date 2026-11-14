"""Financial cost sensitivity analysis and expected business value modeling."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class FinancialImpactReport:
    """Summary of financial and operational impact under a specific decision policy."""
    policy_name: str
    tau_low: float
    tau_high: float
    c_fn_mult: float
    c_fp: float
    c_rev: float
    total_transactions: int
    total_fraud_volume: float
    uncaught_fraud_amount: float
    saved_fraud_amount: float
    fp_friction_cost: float
    review_operational_cost: float
    total_financial_loss: float
    baseline_loss: float
    net_financial_savings: float
    roi_percentage: float
    approved_count: int
    review_count: int
    declined_count: int
    review_capacity_exceeded: bool


class CostAnalyzer:
    """Simulates financial loss, operational triage cost, and customer friction costs."""

    def __init__(
        self,
        false_negative_multiplier: float = 1.0,
        default_false_positive_cost: float = 100.0,
        default_investigation_cost: float = 25.0,
    ) -> None:
        self.fn_mult = false_negative_multiplier
        self.c_fp = default_false_positive_cost
        self.c_rev = default_investigation_cost

    def evaluate_policy(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        amounts: Union[pd.Series, np.ndarray],
        tau_low: float,
        tau_high: float,
        c_fp: Optional[float] = None,
        c_rev: Optional[float] = None,
        max_review_capacity: Optional[int] = None,
        policy_name: str = "TriagePolicy",
    ) -> FinancialImpactReport:
        """Evaluate financial impact under a dual-threshold triage policy:

        - y_prob < tau_low: APPROVE (Straight-Through)
        - tau_low <= y_prob < tau_high: REVIEW (Human Analyst Queue)
        - y_prob >= tau_high: DECLINE (Block / Step-up)
        """
        y_true_arr = np.asarray(y_true, dtype=np.int64)
        y_prob_arr = np.asarray(y_prob, dtype=np.float64)
        amounts_arr = np.asarray(amounts, dtype=np.float64)

        c_fp_val = c_fp if c_fp is not None else self.c_fp
        c_rev_val = c_rev if c_rev is not None else self.c_rev

        n_samples = len(y_true_arr)
        total_fraud_volume = float(np.sum(amounts_arr[y_true_arr == 1]))
        baseline_loss = total_fraud_volume * self.fn_mult

        # Decision masks
        is_declined = (y_prob_arr >= tau_high)
        is_reviewed = (y_prob_arr >= tau_low) & (~is_declined)
        is_approved = (~is_declined) & (~is_reviewed)

        num_declined = int(np.sum(is_declined))
        num_reviewed = int(np.sum(is_reviewed))
        num_approved = int(np.sum(is_approved))

        capacity_exceeded = False
        if max_review_capacity is not None and num_reviewed > max_review_capacity:
            capacity_exceeded = True
            # In real operations, overflow queue is either auto-approved or backlogged
            # Here we record flag and evaluate costs with actual review volume

        # Losses:
        # 1. Uncaught fraud: Frauds in approved bucket are 100% loss
        uncaught_fraud_mask = is_approved & (y_true_arr == 1)
        uncaught_fraud_amount = float(np.sum(amounts_arr[uncaught_fraud_mask])) * self.fn_mult

        # Saved fraud: Frauds in review or decline buckets
        saved_fraud_mask = (~is_approved) & (y_true_arr == 1)
        saved_fraud_amount = float(np.sum(amounts_arr[saved_fraud_mask]))

        # 2. False Positive friction cost: Legitimate transactions that were declined
        fp_declined_mask = is_declined & (y_true_arr == 0)
        fp_friction_cost = float(np.sum(fp_declined_mask)) * c_fp_val

        # 3. Operational investigation cost: All transactions sent to review
        review_operational_cost = float(num_reviewed) * c_rev_val

        # Total net cost under policy
        total_policy_loss = uncaught_fraud_amount + fp_friction_cost + review_operational_cost
        net_savings = baseline_loss - total_policy_loss
        roi = (net_savings / baseline_loss * 100.0) if baseline_loss > 0 else 0.0

        return FinancialImpactReport(
            policy_name=policy_name,
            tau_low=tau_low,
            tau_high=tau_high,
            c_fn_mult=self.fn_mult,
            c_fp=c_fp_val,
            c_rev=c_rev_val,
            total_transactions=n_samples,
            total_fraud_volume=total_fraud_volume,
            uncaught_fraud_amount=uncaught_fraud_amount,
            saved_fraud_amount=saved_fraud_amount,
            fp_friction_cost=fp_friction_cost,
            review_operational_cost=review_operational_cost,
            total_financial_loss=total_policy_loss,
            baseline_loss=baseline_loss,
            net_financial_savings=net_savings,
            roi_percentage=roi,
            approved_count=num_approved,
            review_count=num_reviewed,
            declined_count=num_declined,
            review_capacity_exceeded=capacity_exceeded,
        )

    def run_sensitivity_grid(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        amounts: Union[pd.Series, np.ndarray],
        tau_low: float,
        tau_high: float,
        fp_costs: List[float],
        rev_costs: List[float],
    ) -> pd.DataFrame:
        """Run two-dimensional unit cost sensitivity matrix."""
        records = []
        for c_fp_test in fp_costs:
            for c_rev_test in rev_costs:
                report = self.evaluate_policy(
                    y_true=y_true,
                    y_prob=y_prob,
                    amounts=amounts,
                    tau_low=tau_low,
                    tau_high=tau_high,
                    c_fp=c_fp_test,
                    c_rev=c_rev_test,
                )
                records.append({
                    "c_fp": c_fp_test,
                    "c_rev": c_rev_test,
                    "net_savings": report.net_financial_savings,
                    "total_loss": report.total_financial_loss,
                    "roi_pct": report.roi_percentage,
                    "uncaught_fraud": report.uncaught_fraud_amount,
                    "fp_cost": report.fp_friction_cost,
                    "review_cost": report.review_operational_cost,
                })
        return pd.DataFrame(records)


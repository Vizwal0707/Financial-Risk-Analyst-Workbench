"""Operational threshold optimization under analyst capacity constraints."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from src.evaluation.cost_analysis import CostAnalyzer, FinancialImpactReport


@dataclass
class PolicyThresholds:
    """Selected decision policy thresholds and validation metrics."""
    tau_low: float
    tau_high: float
    best_net_savings: float
    best_roi_pct: float
    review_volume: int
    decline_volume: int
    approval_volume: int
    max_capacity: int
    uncaught_fraud_amount: float
    saved_fraud_amount: float


class DecisionPolicyOptimizer:
    """Optimizes dual triage thresholds (tau_low, tau_high) strictly on validation data.

    Enforces that the volume assigned to the manual review queue does not exceed
    the operational review capacity K_max.
    """

    def __init__(
        self,
        cost_analyzer: Optional[CostAnalyzer] = None,
        daily_capacity: int = 500,
        tau_candidates: Optional[List[float]] = None,
    ) -> None:
        self.cost_analyzer = cost_analyzer or CostAnalyzer()
        self.daily_capacity = daily_capacity
        if tau_candidates is not None:
            self.tau_candidates = sorted(tau_candidates)
        else:
            # Finely spaced candidate thresholds
            self.tau_candidates = [
                0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
            ]

    def calculate_partition_capacity(self, hours: float) -> int:
        """Calculate max allowable reviews based on operational hours."""
        return int(np.floor(self.daily_capacity * (hours / 24.0)))

    def optimize_thresholds(
        self,
        y_true: Union[pd.Series, np.ndarray],
        y_prob: Union[pd.Series, np.ndarray],
        amounts: Union[pd.Series, np.ndarray],
        max_capacity: int,
        c_fp: Optional[float] = None,
        c_rev: Optional[float] = None,
    ) -> PolicyThresholds:
        """Grid search over valid (tau_low, tau_high) pairs with tau_low < tau_high.

        Filters candidates that exceed max_capacity in review volume.
        Selects the pair maximizing net financial savings.
        """
        y_true_arr = np.asarray(y_true, dtype=np.int64)
        y_prob_arr = np.asarray(y_prob, dtype=np.float64)
        amounts_arr = np.asarray(amounts, dtype=np.float64)

        best_savings = -float("inf")
        best_report: Optional[FinancialImpactReport] = None
        best_tau_low = 0.35
        best_tau_high = 0.75

        # Evaluate candidate pairs
        for i, tau_l in enumerate(self.tau_candidates):
            for tau_h in self.tau_candidates[i + 1:]:
                # Quick count of review volume
                in_review = np.sum((y_prob_arr >= tau_l) & (y_prob_arr < tau_h))
                if in_review > max_capacity:
                    continue  # Violates hard operational capacity constraint

                report = self.cost_analyzer.evaluate_policy(
                    y_true=y_true_arr,
                    y_prob=y_prob_arr,
                    amounts=amounts_arr,
                    tau_low=tau_l,
                    tau_high=tau_h,
                    c_fp=c_fp,
                    c_rev=c_rev,
                    max_review_capacity=max_capacity,
                )

                if report.net_financial_savings > best_savings:
                    best_savings = report.net_financial_savings
                    best_report = report
                    best_tau_low = tau_l
                    best_tau_high = tau_h

        # Fallback if no pair satisfied capacity
        if best_report is None:
            # Pick conservative default: review highest scores up to capacity
            sorted_indices = np.argsort(-y_prob_arr)
            cutoff_score = float(y_prob_arr[sorted_indices[min(max_capacity, len(y_prob_arr) - 1)]])
            best_tau_low = cutoff_score
            best_tau_high = 0.95
            best_report = self.cost_analyzer.evaluate_policy(
                y_true=y_true_arr,
                y_prob=y_prob_arr,
                amounts=amounts_arr,
                tau_low=best_tau_low,
                tau_high=best_tau_high,
                c_fp=c_fp,
                c_rev=c_rev,
                max_review_capacity=max_capacity,
            )

        return PolicyThresholds(
            tau_low=best_tau_low,
            tau_high=best_tau_high,
            best_net_savings=best_report.net_financial_savings,
            best_roi_pct=best_report.roi_percentage,
            review_volume=best_report.review_count,
            decline_volume=best_report.declined_count,
            approval_volume=best_report.approved_count,
            max_capacity=max_capacity,
            uncaught_fraud_amount=best_report.uncaught_fraud_amount,
            saved_fraud_amount=best_report.saved_fraud_amount,
        )


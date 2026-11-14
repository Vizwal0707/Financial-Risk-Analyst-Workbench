"""Quantitative risk decision policy engine, triage prioritization, and reason code generator.

Defines:
1. Risk Decision: Analytical classification (APPROVE, REVIEW, DECLINE).
2. Operational Recommendation: Simulated workflow actions (STP_APPROVE, QUEUE_L1_REVIEW,
   BLOCK_TRANSACTION, REQUIRE_STEP_UP_AUTHENTICATION).
3. Risk-Weighted Investigation Priority: Dimensionless queue prioritization heuristic.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    """Encapsulates the operational outcome of risk policy evaluation."""

    decision: str  # Analytical risk classification: APPROVE, REVIEW, DECLINE
    operational_recommendation: str  # STP_APPROVE, QUEUE_L1_REVIEW, BLOCK_TRANSACTION, REQUIRE_STEP_UP_AUTHENTICATION
    action_code: str  # Backwards-compatible alias for operational_recommendation
    risk_tier: str  # LOW, MEDIUM, HIGH
    risk_score: float  # [0.0, 100.0]
    reason_codes: List[str]
    triage_priority: Optional[str] = None  # URGENT, ELEVATED, STANDARD (for REVIEW queue)
    risk_weighted_investigation_priority: Optional[float] = None  # P_cal * ln(1 + amount)
    triage_score: Optional[float] = None  # Backwards-compatible alias
    simulated_recommendation_only: bool = True
    recommendation_disclaimer: str = (
        "Simulated operational recommendation for risk evaluation only; "
        "prohibited from triggering real payment authorization, settlement, or authentication workflows."
    )
    policy_metadata: Dict[str, Any] = field(default_factory=dict)


class DecisionPolicyEngine:
    """Implements dual-threshold financial risk policy and analyst triage routing.

    Threshold Logic:
      calibrated_prob < tau_low:
          -> APPROVE (Straight-Through Processing, minimal friction)
      tau_low <= calibrated_prob < tau_high:
          -> REVIEW (Routed to Level-1 Fraud Analyst queue, capacity bounded)
      calibrated_prob >= tau_high:
          -> DECLINE (Simulated transaction block or step-up authentication challenge)

    Risk-Weighted Investigation Priority:
      Computed as: P_calibrated * ln(1 + amount)
      Units: Dimensionless prioritization index (NOT expected monetary loss).
      Purpose: Monotonically rank-orders manual investigation alerts by balancing
               fraud probability against nominal monetary scale while dampening extreme outliers.
    """

    def __init__(
        self,
        tau_low: float = 0.10,
        tau_high: float = 0.75,
        step_up_threshold: float = 0.90,
        max_daily_capacity: int = 500,
        c_fp: float = 100.0,
        c_rev: float = 25.0,
    ) -> None:
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.step_up_threshold = step_up_threshold
        self.max_daily_capacity = max_daily_capacity
        self.c_fp = c_fp
        self.c_rev = c_rev

    @classmethod
    def load_from_artifacts(
        cls,
        thresholds_path: str = "models/decision_policy_thresholds.json",
        config_path: str = "configs/policy_config.yaml",
    ) -> "DecisionPolicyEngine":
        """Factory initializing policy parameters from audited model artifacts."""
        tau_low = 0.10
        tau_high = 0.75
        step_up_threshold = 0.90
        daily_capacity = 500
        c_fp = 100.0
        c_rev = 25.0

        p = Path(thresholds_path)
        if p.exists():
            try:
                with open(p, "r") as f:
                    data = json.load(f)
                tau_low = float(data.get("tau_low", tau_low))
                tau_high = float(data.get("tau_high", tau_high))
                daily_capacity = int(data.get("max_daily_capacity", daily_capacity))
                logger.info(f"Loaded decision thresholds from {thresholds_path}: tau_low={tau_low}, tau_high={tau_high}")
            except Exception as e:
                logger.warning(f"Could not parse {thresholds_path}: {e}. Using defaults.")

        cfg_p = Path(config_path)
        if cfg_p.exists():
            try:
                import yaml
                with open(cfg_p, "r") as f:
                    cfg = yaml.safe_load(f).get("policy", {})
                costs = cfg.get("cost_assumptions", {})
                c_fp = float(costs.get("false_positive_cost", c_fp))
                c_rev = float(costs.get("investigation_cost", c_rev))
            except Exception as e:
                logger.warning(f"Could not load cost config from {config_path}: {e}")

        return cls(
            tau_low=tau_low,
            tau_high=tau_high,
            step_up_threshold=step_up_threshold,
            max_daily_capacity=daily_capacity,
            c_fp=c_fp,
            c_rev=c_rev,
        )

    def evaluate_decision(
        self,
        calibrated_prob: float,
        amount: float,
        features_dict: Dict[str, float],
        anomaly_score: Optional[float] = None,
        cold_start_applied: bool = False,
    ) -> DecisionResult:
        """Evaluates calibrated fraud probability against risk policy and assigns triage tier.

        Separates:
          1. Risk Decision: APPROVE, REVIEW, DECLINE
          2. Operational Recommendation: STP_APPROVE, QUEUE_L1_REVIEW, BLOCK_TRANSACTION, REQUIRE_STEP_UP_AUTHENTICATION
        """
        prob = float(np.clip(calibrated_prob, 0.0, 1.0))
        risk_score = round(prob * 100.0, 2)
        reasons: List[str] = []

        drain_ratio = features_dict.get("balance_orig_drain_ratio", 0.0)
        is_dest_zero = features_dict.get("is_dest_zero_balance", 0.0)
        is_orig_zero = features_dict.get("is_orig_zero_balance", 0.0)
        type_transfer = features_dict.get("type_TRANSFER", 0.0)
        type_cash_out = features_dict.get("type_CASH_OUT", 0.0)
        type_payment = features_dict.get("type_PAYMENT", 0.0)
        is_merchant = features_dict.get("is_merchant_dest", 0.0)
        c_1h = features_dict.get("orig_tx_count_1h", 0.0)
        c_24h = features_dict.get("orig_tx_count_24h", 0.0)

        # Risk-Weighted Investigation Priority: P_cal * ln(1 + amount)
        # Units: Dimensionless ranking index (NOT expected loss).
        priority_score = round(float(prob * np.log1p(max(amount, 0.0))), 4)

        # 1. Tri-State Risk Decision & Operational Recommendation Mapping
        if prob >= self.tau_high:
            decision = "DECLINE"
            risk_tier = "HIGH"
            triage_priority = "DECLINE"

            # Operational recommendation: Critical risk / severe drain -> BLOCK; High risk step-up candidate -> REQUIRE_STEP_UP
            if prob >= self.step_up_threshold or drain_ratio >= 0.95 or amount >= 500_000.0:
                op_rec = "BLOCK_TRANSACTION"
            else:
                op_rec = "REQUIRE_STEP_UP_AUTHENTICATION"

        elif prob >= self.tau_low:
            decision = "REVIEW"
            op_rec = "QUEUE_L1_REVIEW"
            risk_tier = "MEDIUM"

            if amount >= 500_000.0 or prob >= 0.60:
                triage_priority = "URGENT"
            elif priority_score >= 5.0 or amount >= 100_000.0:
                triage_priority = "ELEVATED"
            else:
                triage_priority = "STANDARD"
        else:
            decision = "APPROVE"
            op_rec = "STP_APPROVE"
            risk_tier = "LOW"
            triage_priority = None

        # 2. Reason-Code Generation with Strict Evidence Grounding
        # A. Balance Drain: rule-derived origin liquidation indicator
        if drain_ratio >= 0.95 and amount > 0.0:
            reasons.append("BALANCE_DRAIN_CRITICAL")

        # B. Destination Mule: applies ONLY to peer customer accounts in TRANSFER/CASH_OUT, NEVER authentic merchants
        if is_dest_zero == 1.0 and is_merchant == 0.0 and (type_transfer == 1.0 or type_cash_out == 1.0):
            reasons.append("DEST_ZERO_BALANCE_MULE")

        # C. Origin Zero Balance: sender initiated transfer with zero prior balance
        if is_orig_zero == 1.0:
            reasons.append("ORIG_ZERO_BALANCE")

        # D. Payment Mechanism Corridors (Type-based, non-geographic)
        if type_transfer == 1.0:
            reasons.append("CORRIDOR_TRANSFER_HIGH_RISK")
        elif type_cash_out == 1.0:
            reasons.append("CORRIDOR_CASH_OUT_HIGH_RISK")

        # E. Nominal Amount Threshold
        if amount >= 200_000.0:
            reasons.append("HIGH_TRANSACTION_AMOUNT")

        # F. Velocity Spikes: ONLY generated when historical context was provided (suppressed during cold start)
        if not cold_start_applied:
            if c_1h > 1.0:
                reasons.append("ELEVATED_VELOCITY_1H")
            if c_24h > 5.0:
                reasons.append("ELEVATED_VELOCITY_24H")

        # G. Deep Autoencoder Reconstruction Anomaly (Model-derived)
        if anomaly_score is not None and anomaly_score >= 0.90:
            reasons.append("ANOMALOUS_AUTOENCODER_RECONSTRUCTION")

        # H. Approval Rationale
        if decision == "APPROVE":
            if is_merchant == 1.0 or type_payment == 1.0:
                reasons.append("LOW_RISK_MERCHANT_PAYMENT")
            elif not reasons:
                reasons.append("STP_NORMAL_ACTIVITY")

        return DecisionResult(
            decision=decision,
            operational_recommendation=op_rec,
            action_code=op_rec,
            risk_tier=risk_tier,
            risk_score=risk_score,
            reason_codes=reasons,
            triage_priority=triage_priority,
            risk_weighted_investigation_priority=priority_score,
            triage_score=priority_score,
            simulated_recommendation_only=True,
            policy_metadata={
                "tau_low": self.tau_low,
                "tau_high": self.tau_high,
                "step_up_threshold": self.step_up_threshold,
                "max_daily_capacity": self.max_daily_capacity,
            },
        )

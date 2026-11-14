"""Model governance and lineage metadata endpoint."""

from fastapi import APIRouter, Request
from src.schemas.response import ModelInfoResponse

router = APIRouter(prefix="/v1/model", tags=["Model Governance"])


@router.get("/info", response_model=ModelInfoResponse, summary="Model Metadata & Policy Configuration")
async def get_model_info(request: Request) -> ModelInfoResponse:
    """Returns metadata, active feature names, calibration parameters, and decision policy thresholds."""
    state = request.app.state
    feature_names = getattr(state, "feature_names", [])
    policy_engine = getattr(state, "policy_engine", None)

    tau_low = policy_engine.tau_low if policy_engine else 0.10
    tau_high = policy_engine.tau_high if policy_engine else 0.75
    daily_cap = policy_engine.max_daily_capacity if policy_engine else 500
    c_fp = policy_engine.c_fp if policy_engine else 100.0
    c_rev = policy_engine.c_rev if policy_engine else 25.0

    calibrator = getattr(state, "calibrator", None)
    cal_method = calibrator.method if calibrator else "none"

    return ModelInfoResponse(
        model_name="PaySim_CostWeightedXGBoost_Calibrated",
        model_version=getattr(state, "model_version", "1.0.0"),
        model_type="Cost-Sensitive Gradient Boosted Trees (XGBoost) + Non-parametric Isotonic Regression",
        n_features=len(feature_names),
        feature_names=feature_names,
        calibration_method=cal_method,
        calibration_status="audited_validation_fitted",
        decision_policy={
            "tau_low": tau_low,
            "tau_high": tau_high,
            "primary_score": "Calibrated_XGBoost",
            "actions": {
                "APPROVE": f"Calibrated Score < {tau_low:.2f} (Straight-Through Processing)",
                "REVIEW": f"{tau_low:.2f} <= Calibrated Score < {tau_high:.2f} (Human Analyst Queue)",
                "DECLINE": f"Calibrated Score >= {tau_high:.2f} (Immediate Block / Step-up MFA)",
            },
            "max_daily_capacity": daily_cap,
        },
        cost_assumptions={
            "false_positive_cost": c_fp,
            "investigation_cost": c_rev,
            "false_negative_multiplier": 1.0,
        },
        audit_disclaimer=(
            "Quantitative research prototype evaluated on the synthetic PaySim benchmark (6,362,620 transactions). "
            "Simulated decision thresholds do not constitute statutory bank authorization decisions and are not "
            "connected to live interbank clearing systems."
        ),
    )


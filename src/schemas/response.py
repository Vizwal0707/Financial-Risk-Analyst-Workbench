"""Response schemas and risk decision contracts for the scoring engine API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FeatureContributionResponse(BaseModel):
    """Individual feature risk attribution item."""

    feature_name: str = Field(..., description="Name of the evaluated feature")
    feature_value: float = Field(..., description="Raw observed value of the feature")
    contribution: float = Field(..., description="SHAP attribution value (log-odds impact)")
    direction: str = Field(..., description="RISK_INDUCING or RISK_MITIGATING")
    plain_english: str = Field(..., description="Human-readable financial risk explanation")


class FraudExplanationResponse(BaseModel):
    """Explainable risk assessment card for Level-1 Fraud Operations."""

    narrative: str = Field(..., description="Holistic plain-English summary of risk rationale")
    top_risk_factors: List[FeatureContributionResponse] = Field(
        default_factory=list,
        description="Top features pushing probability toward fraud",
    )
    top_mitigating_factors: List[FeatureContributionResponse] = Field(
        default_factory=list,
        description="Top features pushing probability toward legitimate",
    )


class FraudScoreResponse(BaseModel):
    """Scoring response for a single transaction evaluation."""

    request_id: str = Field(..., description="Unique correlation identifier for tracing")
    transaction_id: str = Field(..., description="Identifier constructed from origin entity and step")
    raw_probability: float = Field(..., description="Uncalibrated output probability from XGBoost booster")
    calibrated_probability: float = Field(..., description="Isotonically calibrated empirical fraud probability")
    risk_score: float = Field(..., description="Normalized risk score on scale [0.0, 100.0]")
    risk_tier: str = Field(..., description="Risk tier: LOW, MEDIUM, or HIGH")
    decision: str = Field(..., description="Analytical risk decision: APPROVE, REVIEW, or DECLINE")
    risk_decision: Optional[str] = Field(
        default=None,
        description="Explicit risk decision alias: APPROVE, REVIEW, DECLINE (matches decision)",
    )
    operational_recommendation: str = Field(
        ...,
        description="Simulated workflow recommendation: STP_APPROVE, QUEUE_L1_REVIEW, BLOCK_TRANSACTION, REQUIRE_STEP_UP_AUTHENTICATION",
    )
    action_code: str = Field(..., description="Backwards-compatible alias for operational_recommendation")
    calibrated_fraud_probability: Optional[float] = Field(
        default=None,
        description="Calibrated fraud probability alias matching calibrated_probability",
    )
    decision_reasons: List[str] = Field(default_factory=list, description="Rule- and model-driven reason codes")
    risk_weighted_investigation_priority: Optional[float] = Field(
        default=None,
        description="Dimensionless investigation priority index P_cal * ln(1 + amount) for queue ranking (NOT expected loss)",
    )
    triage_score: Optional[float] = Field(
        default=None,
        description="Backwards-compatible alias for risk_weighted_investigation_priority",
    )
    anomaly_score: Optional[float] = Field(default=None, description="Autoencoder reconstruction anomaly percentile in [0.0, 1.0]")
    latency_ms: float = Field(..., description="Inference pipeline execution latency in milliseconds")
    model_version: str = Field(..., description="Scoring model version identifier")
    cold_start_applied: bool = Field(default=False, description="Whether cold-start default history was utilized")
    simulated_recommendation_only: bool = Field(
        default=True,
        description="Safety flag ensuring simulated recommendations are never connected to live payment clearing",
    )
    recommendation_disclaimer: str = Field(
        default="Simulated operational recommendation for risk evaluation only; prohibited from triggering real authorization or settlement.",
        description="Compliance and safety guardrail statement",
    )
    explanation: Optional[FraudExplanationResponse] = Field(default=None, description="SHAP attribution card if requested")


class BatchScoreSummary(BaseModel):
    """Aggregated portfolio summary of a batch evaluation run."""

    total_evaluated: int = Field(..., description="Total transactions scored in batch")
    approvals: int = Field(..., description="Number of straight-through approved transactions")
    reviews: int = Field(..., description="Number of transactions routed to manual investigation")
    declines: int = Field(..., description="Number of transactions blocked / declined")
    total_amount_evaluated: float = Field(..., description="Total nominal dollar amount evaluated")
    total_amount_declined: float = Field(..., description="Total dollar volume blocked under decline policy")
    mean_calibrated_probability: float = Field(..., description="Average calibrated fraud risk in batch")
    mean_latency_ms: float = Field(..., description="Average per-transaction processing latency")


class BatchScoreResponse(BaseModel):
    """Batch scoring response container."""

    batch_id: str = Field(..., description="Unique batch run identifier")
    summary: BatchScoreSummary = Field(..., description="Summary statistics for batch")
    results: List[FraudScoreResponse] = Field(..., description="Individual scoring results for each transaction")
    total_latency_ms: float = Field(..., description="Total wall-clock duration of batch execution in ms")


class ModelInfoResponse(BaseModel):
    """Model governance metadata, lineage, feature roster, and policy parameters."""

    model_name: str = Field(..., description="Formal model asset name")
    model_version: str = Field(..., description="Semantic version of active scoring pipeline")
    model_type: str = Field(..., description="Primary algorithm architecture")
    n_features: int = Field(..., description="Number of input features required")
    feature_names: List[str] = Field(..., description="Strictly ordered list of model feature names")
    calibration_method: str = Field(..., description="Active probability calibration algorithm")
    calibration_status: str = Field(..., description="Calibration audit status")
    decision_policy: Dict[str, Any] = Field(..., description="Active dual thresholds and capacity rules")
    cost_assumptions: Dict[str, Any] = Field(..., description="Financial optimization cost multipliers")
    audit_disclaimer: str = Field(..., description="Compliance and regulatory research disclaimer")


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str = Field(default="healthy", description="Service liveness indicator")
    timestamp: str = Field(..., description="ISO 8601 current timestamp")
    version: str = Field(..., description="API service version")


class ReadinessResponse(BaseModel):
    """Readiness probe response indicating model availability and database status."""

    status: str = Field(..., description="Service readiness status: ready or degraded")
    models_loaded: bool = Field(..., description="Whether primary models and calibrators are loaded in memory")
    database_connected: bool = Field(..., description="Whether audit persistence layer is reachable")
    timestamp: str = Field(..., description="ISO 8601 current timestamp")
    details: Dict[str, Any] = Field(default_factory=dict, description="Component status details")


class ErrorResponse(BaseModel):
    """Standardized error response conforming to RFC 7807 problem details pattern."""

    type: str = Field(..., description="URI or error category identifier")
    title: str = Field(..., description="Brief human-readable summary of the problem")
    status: int = Field(..., description="HTTP status code")
    detail: str = Field(..., description="Detailed explanation specific to this occurrence")
    instance: Optional[str] = Field(default=None, description="Request ID or resource URI associated with error")


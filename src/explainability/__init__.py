"""Model explainability and audit attribution engines."""

from src.explainability.shap_explainer import (
    XGBoostShapExplainer,
    AutoencoderAttributionExplainer,
    FraudExplanationCard,
)

__all__ = [
    "XGBoostShapExplainer",
    "AutoencoderAttributionExplainer",
    "FraudExplanationCard",
]


"""Explainability serving module delivering fast TreeSHAP attribution and audit cards."""

import logging
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.explainability.shap_explainer import XGBoostShapExplainer
from src.schemas.response import (
    FeatureContributionResponse,
    FraudExplanationResponse,
)

logger = logging.getLogger(__name__)


class ExplainerService:
    """Service wrapping TreeSHAP attributions and formatting response schemas."""

    def __init__(self, model: Any, feature_names: List[str]) -> None:
        self.feature_names = feature_names
        self.shap_explainer = XGBoostShapExplainer(model=model, feature_names=feature_names)

    def explain_transaction(
        self,
        features_row: Union[pd.Series, pd.DataFrame, np.ndarray],
        predicted_prob: float,
        decision: str,
        top_n: int = 5,
        anomaly_score: Optional[float] = None,
    ) -> FraudExplanationResponse:
        """Computes local TreeSHAP attributions and returns structured response."""
        card = self.shap_explainer.explain_instance(
            row=features_row,
            predicted_prob=predicted_prob,
            decision=decision,
            top_n=top_n,
            anomaly_score=anomaly_score,
        )

        top_risks = [
            FeatureContributionResponse(
                feature_name=f.feature_name,
                feature_value=round(f.feature_value, 4),
                contribution=round(f.contribution, 4),
                direction=f.direction,
                plain_english=f.plain_english,
            )
            for f in card.top_risk_factors
        ]

        top_mits = [
            FeatureContributionResponse(
                feature_name=f.feature_name,
                feature_value=round(f.feature_value, 4),
                contribution=round(f.contribution, 4),
                direction=f.direction,
                plain_english=f.plain_english,
            )
            for f in card.top_mitigating_factors
        ]

        return FraudExplanationResponse(
            narrative=card.narrative_summary,
            top_risk_factors=top_risks,
            top_mitigating_factors=top_mits,
        )


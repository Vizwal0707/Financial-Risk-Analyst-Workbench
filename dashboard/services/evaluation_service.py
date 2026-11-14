"""Service for loading offline audited model evaluation benchmarks and reports."""

import json
import logging
import os
from typing import Any, Dict, Optional
import pandas as pd

from dashboard.config.settings import settings

logger = logging.getLogger(__name__)


class EvaluationDataService:
    """Provides structured access to frozen Phase 4 / 4.1 model metrics and benchmark artifacts."""

    def __init__(self) -> None:
        self.comparison_path = settings.model_comparison_path
        self.test_metrics_path = settings.test_metrics_path
        self.confusion_matrix_path = settings.confusion_matrix_path
        self.ablation_path = settings.feature_ablation_path

    def get_model_comparison(self) -> Dict[str, Any]:
        """Loads validation benchmark comparisons across models from model_comparison.json."""
        if not os.path.exists(self.comparison_path):
            logger.warning(f"File not found: {self.comparison_path}")
            return {}
        try:
            with open(self.comparison_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {self.comparison_path}: {e}")
            return {}

    def get_test_metrics(self) -> Dict[str, Any]:
        """Loads out-of-time frozen test metrics from test_evaluation_metrics.json."""
        if not os.path.exists(self.test_metrics_path):
            logger.warning(f"File not found: {self.test_metrics_path}")
            return {}
        try:
            with open(self.test_metrics_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {self.test_metrics_path}: {e}")
            return {}

    def get_confusion_matrix(self) -> pd.DataFrame:
        """Loads operational 3x3 confusion matrix from decision_confusion_matrix.csv."""
        if not os.path.exists(self.confusion_matrix_path):
            logger.warning(f"File not found: {self.confusion_matrix_path}")
            return pd.DataFrame()
        try:
            return pd.read_csv(self.confusion_matrix_path)
        except Exception as e:
            logger.error(f"Failed to read {self.confusion_matrix_path}: {e}")
            return pd.DataFrame()

    def get_feature_ablation_results(self) -> pd.DataFrame:
        """Loads 11 feature ablation experiments from feature_ablation_results.csv."""
        if not os.path.exists(self.ablation_path):
            logger.warning(f"File not found: {self.ablation_path}")
            return pd.DataFrame()
        try:
            return pd.read_csv(self.ablation_path)
        except Exception as e:
            logger.error(f"Failed to read {self.ablation_path}: {e}")
            return pd.DataFrame()

    def get_drift_status(self) -> Dict[str, Any]:
        """Returns regulatory drift monitoring status (formally documented as unavailable)."""
        return {
            "status": "UNAVAILABLE",
            "reason": (
                "The PaySim evaluation benchmark contains a frozen 743-hour simulation horizon with no ongoing "
                "continuous streaming reference data. Synthetic drift calculation is rejected to preserve audit integrity."
            ),
            "recommendation": "Deploy real-time PSI/KS divergence monitor when streaming live production transactions.",
        }


evaluation_service = EvaluationDataService()


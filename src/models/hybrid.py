"""Hybrid fraud detection model combining supervised XGBoost and unsupervised Autoencoder."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import joblib


class HybridFraudScorer:
    """Ensembles calibrated supervised XGBoost probabilities with unsupervised Autoencoder anomaly scores.

    Formula:
        S_hybrid = alpha * P_XGB + (1 - alpha) * S_AE

    where P_XGB in [0, 1] is calibrated probability, and S_AE in [0, 1] is normalized
    reconstruction error CDF percentile.
    """

    def __init__(self, alpha: float = 0.80) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in [0.0, 1.0], got {alpha}")
        self.alpha = alpha
        self.best_alpha_: float = alpha
        self.best_val_pr_auc_: float = 0.0

    def compute_score(
        self,
        p_xgb: Union[pd.Series, np.ndarray],
        s_ae: Union[pd.Series, np.ndarray],
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Compute hybrid risk score from component predictions."""
        a = self.alpha if alpha is None else alpha
        p = np.clip(np.asarray(p_xgb, dtype=np.float64), 0.0, 1.0)
        s = np.clip(np.asarray(s_ae, dtype=np.float64), 0.0, 1.0)
        return np.clip(a * p + (1.0 - a) * s, 0.0, 1.0)

    def optimize_alpha(
        self,
        p_xgb_val: Union[pd.Series, np.ndarray],
        s_ae_val: Union[pd.Series, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        alpha_grid: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Grid search optimal alpha on validation partition using PR-AUC criterion."""
        if alpha_grid is None:
            alpha_grid = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

        y_true = np.asarray(y_val, dtype=np.int64)
        best_auc = -1.0
        best_a = self.alpha
        grid_results = {}

        for a in alpha_grid:
            score = self.compute_score(p_xgb_val, s_ae_val, alpha=a)
            auc = float(average_precision_score(y_true, score))
            grid_results[f"alpha_{a:.2f}"] = auc
            if auc > best_auc:
                best_auc = auc
                best_a = a

        self.best_alpha_ = best_a
        self.alpha = best_a
        self.best_val_pr_auc_ = best_auc

        # Determine if hybrid genuinely outperforms standalone XGBoost (alpha=1.0)
        xgb_standalone_auc = grid_results.get("alpha_1.00", 0.0)
        ae_standalone_auc = grid_results.get("alpha_0.00", 0.0)
        hybrid_advantage = best_auc - xgb_standalone_auc

        return {
            "best_alpha": self.best_alpha_,
            "best_val_pr_auc": self.best_val_pr_auc_,
            "xgb_standalone_pr_auc": xgb_standalone_auc,
            "ae_standalone_pr_auc": ae_standalone_auc,
            "hybrid_outperforms_standalone": (hybrid_advantage > 1e-4),
            "delta_pr_auc_over_xgb": float(hybrid_advantage),
            "grid_results": grid_results,
        }

    def save(self, filepath: str) -> None:
        """Persist hybrid configuration."""
        joblib.dump({"alpha": self.alpha, "best_alpha": self.best_alpha_}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "HybridFraudScorer":
        """Load hybrid configuration."""
        data = joblib.load(filepath)
        instance = cls(alpha=data["alpha"])
        instance.best_alpha_ = data.get("best_alpha", data["alpha"])
        return instance


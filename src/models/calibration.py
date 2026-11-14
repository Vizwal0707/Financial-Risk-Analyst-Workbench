"""Probability calibration module for fraud classification models.

Implements:
1. Sigmoid Calibration (Platt Scaling via Logistic Regression on log-odds)
2. Non-parametric Isotonic Regression Calibration
3. Calibration diagnostics (Brier score, ECE, MCE, reliability curves)
Strictly fitted on validation partition predictions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import joblib

from src.evaluation.metrics import calculate_brier_and_ece


class ProbabilityCalibrator:
    """Calibrates model output scores into empirical probabilities."""

    def __init__(self, method: str = "isotonic") -> None:
        if method not in ["sigmoid", "isotonic"]:
            raise ValueError(f"Method must be 'sigmoid' or 'isotonic', got '{method}'")
        self.method = method
        self.calibrator_: Optional[Union[LogisticRegression, IsotonicRegression]] = None
        self.is_fitted_: bool = False

    def fit(
        self,
        uncalibrated_probs: Union[pd.Series, np.ndarray],
        y_true: Union[pd.Series, np.ndarray],
    ) -> "ProbabilityCalibrator":
        """Fit calibration model strictly on validation predictions and labels."""
        p_raw = np.clip(np.asarray(uncalibrated_probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        y = np.asarray(y_true, dtype=np.int64)

        if self.method == "sigmoid":
            # Platt Scaling: Logistic Regression on logit (log-odds)
            logits = np.log(p_raw / (1.0 - p_raw)).reshape(-1, 1)
            clf = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
            clf.fit(logits, y)
            self.calibrator_ = clf
        elif self.method == "isotonic":
            # Isotonic Regression: non-decreasing step function
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(p_raw, y)
            self.calibrator_ = iso

        self.is_fitted_ = True
        return self

    def calibrate(self, uncalibrated_probs: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Transform raw uncalibrated probabilities into calibrated probabilities."""
        if not self.is_fitted_ or self.calibrator_ is None:
            raise RuntimeError("ProbabilityCalibrator must be fitted before calling calibrate()")

        p_raw = np.clip(np.asarray(uncalibrated_probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)

        if self.method == "sigmoid":
            logits = np.log(p_raw / (1.0 - p_raw)).reshape(-1, 1)
            calibrated = self.calibrator_.predict_proba(logits)[:, 1]
        elif self.method == "isotonic":
            calibrated = self.calibrator_.predict(p_raw)

        return np.clip(calibrated, 0.0, 1.0)

    @classmethod
    def compare_methods(
        cls,
        uncalibrated_probs: Union[pd.Series, np.ndarray],
        y_true: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Dict[str, Any]]:
        """Fit both Sigmoid and Isotonic calibrators on validation data and return comparison metrics."""
        y_arr = np.asarray(y_true, dtype=np.int64)
        p_arr = np.asarray(uncalibrated_probs, dtype=np.float64)

        uncal_metrics = calculate_brier_and_ece(y_arr, p_arr)

        # Sigmoid (Platt)
        cal_sig = cls(method="sigmoid").fit(p_arr, y_arr)
        sig_probs = cal_sig.calibrate(p_arr)
        sig_metrics = calculate_brier_and_ece(y_arr, sig_probs)

        # Isotonic
        cal_iso = cls(method="isotonic").fit(p_arr, y_arr)
        iso_probs = cal_iso.calibrate(p_arr)
        iso_metrics = calculate_brier_and_ece(y_arr, iso_probs)

        return {
            "uncalibrated": {
                "brier_score": uncal_metrics["brier_score"],
                "ece": uncal_metrics["expected_calibration_error"],
                "mce": uncal_metrics["maximum_calibration_error"],
            },
            "sigmoid": {
                "brier_score": sig_metrics["brier_score"],
                "ece": sig_metrics["expected_calibration_error"],
                "mce": sig_metrics["maximum_calibration_error"],
                "calibrator": cal_sig,
            },
            "isotonic": {
                "brier_score": iso_metrics["brier_score"],
                "ece": iso_metrics["expected_calibration_error"],
                "mce": iso_metrics["maximum_calibration_error"],
                "calibrator": cal_iso,
            },
        }

    def save(self, filepath: str) -> None:
        """Persist fitted calibrator to disk."""
        joblib.dump({"method": self.method, "calibrator": self.calibrator_}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ProbabilityCalibrator":
        """Load fitted calibrator from disk."""
        data = joblib.load(filepath)
        instance = cls(method=data["method"])
        instance.calibrator_ = data["calibrator"]
        instance.is_fitted_ = True
        return instance


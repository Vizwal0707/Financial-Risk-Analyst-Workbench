"""Supervised XGBoost fraud classifier with cost-weighted loss and PR-AUC early stopping."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = object  # type: ignore


class FraudXGBoostClassifier:
    """Enterprise-grade XGBoost classifier tuned for extreme fraud class imbalance."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_child_weight: float = 5.0,
        subsample: float = 0.80,
        colsample_bytree: float = 0.80,
        n_estimators: int = 500,
        early_stopping_rounds: int = 30,
        scale_pos_weight: Optional[float] = None,
        tree_method: str = "hist",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for FraudXGBoostClassifier.")

        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.scale_pos_weight = scale_pos_weight
        self.tree_method = tree_method
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model: Optional[XGBClassifier] = None
        self.feature_names_: List[str] = []
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "FraudXGBoostClassifier":
        """Train XGBoost model with PR-AUC early stopping on validation partition."""
        if feature_names is not None:
            self.feature_names_ = feature_names
        elif isinstance(X_train, pd.DataFrame):
            self.feature_names_ = list(X_train.columns)
        else:
            self.feature_names_ = [f"f_{i}" for i in range(X_train.shape[1])]

        y_train_arr = np.asarray(y_train, dtype=np.int64)

        # Calculate scale_pos_weight dynamically if not provided
        if self.scale_pos_weight is None:
            n_pos = np.sum(y_train_arr == 1)
            n_neg = np.sum(y_train_arr == 0)
            self.scale_pos_weight = float(n_neg / max(n_pos, 1))

        self.model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_estimators=self.n_estimators,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            scale_pos_weight=self.scale_pos_weight,
            tree_method=self.tree_method,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val[self.feature_names_], np.asarray(y_val, dtype=np.int64))]

        self.model.fit(
            X_train[self.feature_names_],
            y_train_arr,
            eval_set=eval_set,
            verbose=False,
        )

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities [P(legit), P(fraud)]."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return self.model.predict_proba(X[self.feature_names_])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class given custom decision threshold."""
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(np.int64)

    def get_feature_importances(self) -> pd.DataFrame:
        """Return feature importance metrics sorted by gain."""
        if self.model is None:
            raise RuntimeError("Model is not fitted.")

        booster = self.model.get_booster()
        score_gain = booster.get_score(importance_type="gain")
        score_weight = booster.get_score(importance_type="weight")
        score_cover = booster.get_score(importance_type="cover")

        records = []
        for feat in self.feature_names_:
            records.append({
                "feature": feat,
                "importance_gain": float(score_gain.get(feat, 0.0)),
                "importance_weight": float(score_weight.get(feat, 0.0)),
                "importance_cover": float(score_cover.get(feat, 0.0)),
            })

        df = pd.DataFrame(records).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)
        return df

    def save(self, filepath: str) -> None:
        """Serialize trained model and feature schema."""
        if self.model is None:
            raise RuntimeError("Cannot save un-fitted model.")
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names_,
                "scale_pos_weight": self.scale_pos_weight,
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str) -> "FraudXGBoostClassifier":
        """Load serialized model from disk."""
        data = joblib.load(filepath)
        instance = cls()
        instance.model = data["model"]
        instance.feature_names_ = data["feature_names"]
        instance.scale_pos_weight = data.get("scale_pos_weight")
        return instance


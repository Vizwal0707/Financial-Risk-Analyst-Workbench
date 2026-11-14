"""Baseline fraud classification models for benchmark comparison.

Includes:
1. MajorityClassClassifier (predicts legitimate class zero)
2. TransactionTypeClassifier (empirical corridor fraud rate)
3. RuleHeuristicClassifier (domain-specific rule engine)
4. LogisticRegressionBaseline (linear benchmark with robust scaling)
5. RandomForestBaseline (non-linear bagged ensemble benchmark)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import joblib


class MajorityClassClassifier:
    """Predicts majority class (0 / legitimate) for all transactions."""

    def __init__(self) -> None:
        self.prior_probability_: float = 0.0
        self.classes_ = np.array([0, 1])

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "MajorityClassClassifier":
        y_arr = np.asarray(y)
        self.prior_probability_ = float(np.mean(y_arr))
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        n_samples = len(X)
        proba_1 = np.full(n_samples, self.prior_probability_, dtype=np.float64)
        proba_0 = 1.0 - proba_1
        return np.column_stack([proba_0, proba_1])

    def predict(self, X: Union[pd.DataFrame, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        n_samples = len(X)
        return np.zeros(n_samples, dtype=np.int64)


class TransactionTypeClassifier:
    """Predicts empirical fraud probability conditioned solely on transaction corridor type.

    In PaySim, fraud is strictly isolated to TRANSFER and CASH_OUT typologies.
    This baseline quantifies how much signal originates purely from corridor choice.
    """

    def __init__(self) -> None:
        self.type_fraud_rates_: Dict[str, float] = {}
        self.global_fraud_rate_: float = 0.0
        self.classes_ = np.array([0, 1])

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        type_column: str = "type",
    ) -> "TransactionTypeClassifier":
        y_arr = np.asarray(y)
        self.global_fraud_rate_ = float(np.mean(y_arr))

        if type_column in X.columns:
            types = X[type_column].astype(str)
            unique_types = types.unique()
            for t in unique_types:
                mask = (types == t)
                rate = float(np.mean(y_arr[mask])) if mask.sum() > 0 else self.global_fraud_rate_
                self.type_fraud_rates_[t] = rate
        else:
            # Check for one-hot encoded type columns
            for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
                col = f"type_{t}"
                if col in X.columns:
                    mask = (X[col] == 1)
                    rate = float(np.mean(y_arr[mask])) if mask.sum() > 0 else 0.0
                    self.type_fraud_rates_[t] = rate

        return self

    def predict_proba(self, X: pd.DataFrame, type_column: str = "type") -> np.ndarray:
        n_samples = len(X)
        proba_1 = np.full(n_samples, self.global_fraud_rate_, dtype=np.float64)

        if type_column in X.columns:
            types = X[type_column].astype(str).values
            for t, rate in self.type_fraud_rates_.items():
                proba_1[types == t] = rate
        else:
            for t, rate in self.type_fraud_rates_.items():
                col = f"type_{t}"
                if col in X.columns:
                    proba_1[X[col].values == 1] = rate

        proba_0 = 1.0 - proba_1
        return np.column_stack([proba_0, proba_1])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(np.int64)


class RuleHeuristicClassifier:
    """Domain rule engine representing conventional bank rules:

    Rule 1: Large Transfer Rule -> TRANSFER corridor AND amount >= amount_threshold ($200,000)
    Rule 2: Account Drain Rule -> TRANSFER corridor AND balance_orig_drain_ratio >= 0.95
    """

    def __init__(
        self,
        amount_threshold: float = 200000.0,
        drain_ratio_threshold: float = 0.95,
        flag_score: float = 0.95,
        unflagged_score: float = 0.001,
    ) -> None:
        self.amount_threshold = amount_threshold
        self.drain_ratio_threshold = drain_ratio_threshold
        self.flag_score = flag_score
        self.unflagged_score = unflagged_score
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None) -> "RuleHeuristicClassifier":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n_samples = len(X)
        proba_1 = np.full(n_samples, self.unflagged_score, dtype=np.float64)

        # Identify TRANSFER corridor
        if "type" in X.columns:
            is_transfer = (X["type"] == "TRANSFER")
        elif "type_TRANSFER" in X.columns:
            is_transfer = (X["type_TRANSFER"] == 1)
        else:
            is_transfer = pd.Series(False, index=X.index)

        # Condition 1: Large amount
        amount = X["amount"] if "amount" in X.columns else np.exp(X.get("amount_log", 0)) - 1.0
        large_amount = (amount >= self.amount_threshold)

        # Condition 2: Account drain
        if "balance_orig_drain_ratio" in X.columns:
            drain = (X["balance_orig_drain_ratio"] >= self.drain_ratio_threshold)
        else:
            drain = pd.Series(False, index=X.index)

        rule_triggered = is_transfer & (large_amount | drain)
        proba_1[rule_triggered.values] = self.flag_score
        proba_0 = 1.0 - proba_1
        return np.column_stack([proba_0, proba_1])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(np.int64)


class LogisticRegressionBaseline:
    """Class-weighted regularized Logistic Regression benchmark."""

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        class_weight: str = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.scaler = RobustScaler()
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs",
        )
        self.feature_names_: List[str] = []
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "LogisticRegressionBaseline":
        self.feature_names_ = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.fillna(0.0))
        self.model.fit(X_scaled, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X[self.feature_names_].fillna(0.0))
        return self.model.predict_proba(X_scaled)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(np.int64)


class RandomForestBaseline:
    """Balanced Random Forest benchmark."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.feature_names_: List[str] = []
        self.classes_ = np.array([0, 1])

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> "RandomForestBaseline":
        self.feature_names_ = list(X.columns)
        self.model.fit(X[self.feature_names_].fillna(0.0), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names_].fillna(0.0))

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(np.int64)


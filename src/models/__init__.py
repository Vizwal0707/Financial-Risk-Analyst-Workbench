"""Model implementations for financial fraud detection.

Includes baseline heuristics, supervised gradient boosting, deep unsupervised autoencoder,
calibration wrappers, and hybrid decision fusion.
"""

from src.models.baselines import (
    MajorityClassClassifier,
    TransactionTypeClassifier,
    RuleHeuristicClassifier,
    LogisticRegressionBaseline,
    RandomForestBaseline,
)
from src.models.autoencoder import DeepAutoencoder, AutoencoderTrainer
from src.models.xgboost_model import FraudXGBoostClassifier
from src.models.calibration import ProbabilityCalibrator
from src.models.hybrid import HybridFraudScorer

__all__ = [
    "MajorityClassClassifier",
    "TransactionTypeClassifier",
    "RuleHeuristicClassifier",
    "LogisticRegressionBaseline",
    "RandomForestBaseline",
    "DeepAutoencoder",
    "AutoencoderTrainer",
    "FraudXGBoostClassifier",
    "ProbabilityCalibrator",
    "HybridFraudScorer",
]


"""Lifespan manager for preloading machine learning models, calibrators, and database pools."""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI
import joblib

from src.decision_engine.policy_engine import DecisionPolicyEngine
from src.persistence.database import DatabaseManager
from src.persistence.audit_repository import AuditRepository
from src.serving.explainer_service import ExplainerService
from src.serving.feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Preloads all model artifacts, calibrators, and DB connections into app state.

    Ensures zero model reloading on individual request invocations.
    """
    logger.info("Initializing Financial Fraud Scoring Engine lifespan...")

    # 1. Feature Builder
    fb = FeatureBuilder()
    app.state.feature_builder = fb

    # 2. Primary XGBoost Model
    xgb_path = Path("models/xgboost_best.joblib")
    if xgb_path.exists():
        d = joblib.load(xgb_path)
        app.state.xgb_model = d["model"]
        app.state.feature_names = d.get("feature_names", fb.feature_names)
        app.state.model_version = "1.0.0"
        logger.info(f"Loaded primary XGBoost model ({len(app.state.feature_names)} features) from {xgb_path}")
    else:
        logger.error(f"XGBoost model checkpoint not found at: {xgb_path}")
        app.state.xgb_model = None
        app.state.feature_names = fb.feature_names
        app.state.model_version = "1.0.0-unloaded"

    # 3. Probability Calibrator
    cal_path = Path("models/calibrator_best.joblib")
    if cal_path.exists():
        from src.models.calibration import ProbabilityCalibrator
        app.state.calibrator = ProbabilityCalibrator.load(str(cal_path))
        logger.info(f"Loaded probability calibrator ({app.state.calibrator.method}) from {cal_path}")
    else:
        logger.error(f"Calibrator checkpoint not found at: {cal_path}")
        app.state.calibrator = None

    # 4. Deep Autoencoder (Optional Anomaly Signal)
    ae_path = Path("models/autoencoder_best.pt")
    scaler_path = Path("models/autoencoder_scaler.joblib")
    if ae_path.exists() and scaler_path.exists():
        try:
            from src.models.autoencoder import DeepAutoencoder
            app.state.ae_scaler = joblib.load(scaler_path)
            app.state.ae_model = DeepAutoencoder.load(str(ae_path))
            logger.info("Loaded PyTorch Deep Autoencoder and scaler for anomaly scoring.")
        except Exception as e:
            logger.warning(f"Could not load Autoencoder: {e}. Anomaly signal disabled.")
            app.state.ae_model = None
            app.state.ae_scaler = None
    else:
        app.state.ae_model = None
        app.state.ae_scaler = None

    # 5. TreeSHAP Explainer Service
    if app.state.xgb_model is not None:
        try:
            app.state.explainer_service = ExplainerService(
                model=app.state.xgb_model,
                feature_names=app.state.feature_names,
            )
            logger.info("Initialized TreeSHAP ExplainerService.")
        except Exception as e:
            logger.warning(f"TreeSHAP initialization failed: {e}. Explainability in fallback mode.")
            app.state.explainer_service = None
    else:
        app.state.explainer_service = None

    # 6. Decision Policy Engine
    app.state.policy_engine = DecisionPolicyEngine.load_from_artifacts()

    # 7. Persistence & Audit Repository
    db_mgr = DatabaseManager()
    db_mgr.initialize_schema()
    app.state.db_manager = db_mgr
    app.state.audit_repo = AuditRepository(db_mgr)
    logger.info("Audit persistence layer ready.")

    yield

    logger.info("Shutting down Financial Fraud Scoring Engine lifespan...")
    db_mgr = getattr(app.state, "db_manager", None)
    if db_mgr is not None:
        db_mgr.close()
        logger.info("DatabaseManager closed successfully on application shutdown.")


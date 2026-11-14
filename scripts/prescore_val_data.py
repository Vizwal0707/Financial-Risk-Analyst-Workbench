"""Pre-score validation dataset for Policy Simulator exploration.

Ensures that institutional threshold sweeps and capacity simulation operate on the
validation temporal partition (Steps 501–620, N=209,984), preserving the out-of-time
test partition (Steps 621–743, N=90,829) strictly frozen for audit verification.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.models.autoencoder import DeepAutoencoder
from src.models.calibration import ProbabilityCalibrator
from src.models.hybrid import HybridFraudScorer
from src.models.xgboost_model import FraudXGBoostClassifier


def main() -> None:
    val_path = Path("data/processed/val.parquet")
    out_path = Path("data/processed/val_evaluation_scored.parquet")
    models_dir = Path("models")

    print(f"Loading validation partition: {val_path} ...")
    df_val = pd.read_parquet(val_path)
    print(f"Loaded {len(df_val):,} validation rows.")

    forbidden = {
        "step", "type", "nameOrig", "nameDest", "isFraud", "isFlaggedFraud",
        "newbalanceOrig", "newbalanceDest",
    }
    feature_cols = [c for c in df_val.columns if c not in forbidden]
    X_val = df_val[feature_cols].fillna(0.0)

    # Load models
    print("Loading trained model artifacts...")
    xgb = FraudXGBoostClassifier.load(str(models_dir / "xgboost_best.joblib"))
    calibrator = ProbabilityCalibrator.load(str(models_dir / "calibrator_best.joblib"))
    ae_scaler = joblib.load(models_dir / "autoencoder_scaler.joblib")
    ae_model = DeepAutoencoder.load(str(models_dir / "autoencoder_best.pt"))
    hybrid_scorer = HybridFraudScorer.load(str(models_dir / "hybrid_scorer_best.joblib"))

    with open(models_dir / "decision_policy_thresholds.json") as f:
        policy_meta = json.load(f)
    best_alpha = policy_meta.get("best_alpha", 0.95)

    print("Scoring validation partition with XGBoost, Calibrator, and DeepAutoencoder...")
    raw_prob = xgb.predict_proba(X_val)[:, 1].astype(np.float32)
    cal_prob = calibrator.calibrate(raw_prob).astype(np.float32)

    X_val_scaled = ae_scaler.transform(X_val)
    recon_errors, _ = ae_model.compute_reconstruction_errors(X_val_scaled)
    anomaly_score = ae_model.predict_anomaly_score(recon_errors).astype(np.float32)

    primary_score = hybrid_scorer.compute_score(cal_prob, anomaly_score, alpha=best_alpha).astype(np.float32)

    scored_df = pd.DataFrame({
        "step": df_val["step"].values,
        "type": df_val["type"].values,
        "amount": df_val["amount"].values,
        "nameOrig": df_val["nameOrig"].values,
        "nameDest": df_val["nameDest"].values,
        "oldbalanceOrg": df_val["oldbalanceOrg"].values,
        "oldbalanceDest": df_val["oldbalanceDest"].values,
        "isFraud": df_val["isFraud"].values,
        "raw_probability": raw_prob,
        "calibrated_probability": cal_prob,
        "anomaly_score": anomaly_score,
        "primary_score": primary_score,
    })

    scored_df.to_parquet(out_path, index=False)
    print(f"Successfully saved {len(scored_df):,} scored rows to {out_path}")


if __name__ == "__main__":
    main()

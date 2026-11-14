import json
import pandas as pd
import numpy as np
import joblib
from src.evaluation.cost_analysis import CostAnalyzer
from src.models.xgboost_model import FraudXGBoostClassifier
from src.models.calibration import ProbabilityCalibrator
from src.models.hybrid import HybridFraudScorer
from src.models.autoencoder import DeepAutoencoder

df_test = pd.read_parquet("data/processed/test.parquet")
y_test = df_test["isFraud"].values
amounts_test = df_test["amount"].values

with open("models/decision_policy_thresholds.json") as f:
    policy_meta = json.load(f)

tau_low = policy_meta["tau_low"]
tau_high = policy_meta["tau_high"]
best_alpha = policy_meta["best_alpha"]

feature_cols = [c for c in df_test.columns if c not in ["isFraud", "isFlaggedFraud", "step", "partition", "nameOrig", "nameDest", "type"]]
X_test = df_test[feature_cols].fillna(0.0)

xgb = FraudXGBoostClassifier.load("models/xgboost_best.joblib")
cal = ProbabilityCalibrator.load("models/calibrator_best.joblib")
ae_scaler = joblib.load("models/autoencoder_scaler.joblib")
ae_model = DeepAutoencoder.load("models/autoencoder_best.pt")
hybrid_scorer = HybridFraudScorer.load("models/hybrid_scorer_best.joblib")

p_xgb = cal.calibrate(xgb.predict_proba(X_test)[:, 1])
s_ae = ae_model.predict_anomaly_score(ae_model.compute_reconstruction_errors(ae_scaler.transform(X_test))[0])
s_hybrid = hybrid_scorer.compute_score(p_xgb, s_ae, alpha=best_alpha)

cost_analyzer = CostAnalyzer()
fp_costs = [20.0, 50.0, 100.0, 200.0, 500.0]
rev_costs = [10.0, 25.0, 50.0, 75.0, 100.0]

grid = cost_analyzer.run_sensitivity_grid(
    y_true=y_test,
    y_prob=s_hybrid,
    amounts=amounts_test,
    tau_low=tau_low,
    tau_high=tau_high,
    fp_costs=fp_costs,
    rev_costs=rev_costs,
)

pivot = grid.pivot(index="c_fp", columns="c_rev", values="net_savings")
print("=== PROGRAMMATIC RECALCULATION: NET FINANCIAL SAVINGS ($) ===")
print(pivot.map(lambda x: f"${x:,.2f}"))

# Compute investment costs and true ROI
grid["investment_cost"] = grid["c_fp"] * 15 + grid["c_rev"] * 52
grid["true_financial_roi_x"] = grid["net_savings"] / grid["investment_cost"]

pivot_true_roi = grid.pivot(index="c_fp", columns="c_rev", values="true_financial_roi_x")
print("\n=== DEFUSED / DEFENSIBLE METRIC: TRUE FINANCIAL ROI (Net Savings / Operational Costs) ===")
print(pivot_true_roi.map(lambda x: f"{x:,.1f}x"))

pivot_reduction = grid.pivot(index="c_fp", columns="c_rev", values="roi_percentage")
print("\n=== FRAUD LOSS REDUCTION RATE (%) [Previously mislabeled 'ROI'] ===")
print(pivot_reduction.map(lambda x: f"{x:.6f}%"))


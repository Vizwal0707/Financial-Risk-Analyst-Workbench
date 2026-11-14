import json
import duckdb
import joblib
import pandas as pd
import numpy as np
from src.models.xgboost_model import FraudXGBoostClassifier
from src.models.calibration import ProbabilityCalibrator
from src.models.hybrid import HybridFraudScorer
from src.models.autoencoder import DeepAutoencoder

df_test = pd.read_parquet('data/processed/test.parquet')
y_test = df_test['isFraud'].values
amounts_test = df_test['amount'].values

with open('models/decision_policy_thresholds.json') as f:
    policy_meta = json.load(f)

tau_low = policy_meta['tau_low']
tau_high = policy_meta['tau_high']
best_alpha = policy_meta['best_alpha']

feature_cols = [c for c in df_test.columns if c not in ['isFraud', 'isFlaggedFraud', 'step', 'partition', 'nameOrig', 'nameDest', 'type']]
X_test = df_test[feature_cols].fillna(0.0)

xgb_classifier = FraudXGBoostClassifier.load('models/xgboost_best.joblib')
calibrator = ProbabilityCalibrator.load('models/calibrator_best.joblib')

ae_scaler = joblib.load('models/autoencoder_scaler.joblib')
ae_model = DeepAutoencoder.load('models/autoencoder_best.pt')
hybrid_scorer = HybridFraudScorer.load('models/hybrid_scorer_best.joblib')

p_xgb_test_raw = xgb_classifier.predict_proba(X_test)[:, 1]
p_xgb_test_cal = calibrator.calibrate(p_xgb_test_raw)

X_test_scaled = ae_scaler.transform(X_test)
test_recon_errors, _ = ae_model.compute_reconstruction_errors(X_test_scaled)
s_ae_test = ae_model.predict_anomaly_score(test_recon_errors)

s_hybrid_test = hybrid_scorer.compute_score(p_xgb_test_cal, s_ae_test, alpha=best_alpha)

primary_score = s_hybrid_test if policy_meta.get('primary_score') == 'Hybrid_Fusion' else p_xgb_test_cal

print(f"Primary score selected: {policy_meta.get('primary_score')}")
print(f"tau_low: {tau_low:.4f}, tau_high: {tau_high:.4f}")

# Compute decisions
decisions = np.where(primary_score >= tau_high, 'DECLINE', np.where(primary_score >= tau_low, 'REVIEW', 'APPROVE'))
df_test['decision'] = decisions
df_test['primary_score'] = primary_score

ct = pd.crosstab(df_test['isFraud'], df_test['decision'], margins=True)
print("\n=== EXACT 3-WAY DECISION CONFUSION MATRIX (TEST SET) ===")
print(ct)

# Amounts crosstab
amt_ct = df_test.pivot_table(index='isFraud', columns='decision', values='amount', aggfunc='sum', margins=True, fill_value=0.0)
print("\n=== AMOUNT ALLOCATION MATRIX ($) ===")
print(amt_ct.applymap(lambda x: f"${x:,.2f}"))

print("\n--- DETAILED CELL COUNTS AND AMOUNTS ---")
for is_f in [0, 1]:
    lbl = "Legitimate (isFraud=0)" if is_f == 0 else "Fraud (isFraud=1)"
    sub = df_test[df_test['isFraud'] == is_f]
    print(f"\n{lbl} [Total: {len(sub):,} txns, ${sub['amount'].sum():,.2f}]:")
    for d in ['APPROVE', 'REVIEW', 'DECLINE']:
        n = (sub['decision'] == d).sum()
        amt = sub[sub['decision'] == d]['amount'].sum()
        pct_n = n / len(sub) * 100
        pct_amt = (amt / sub['amount'].sum() * 100) if sub['amount'].sum() > 0 else 0
        print(f"  {d:7s}: {n:6,d} txns ({pct_n:6.2f}%) | ${amt:14,.2f} ({pct_amt:6.2f}%)")

# Compare with binary threshold at 0.5:
print("\n--- BINARY METRICS COMPARISON (Threshold = 0.5) ---")
bin_pred_cal = (p_xgb_test_cal >= 0.5).astype(int)
ct_bin = pd.crosstab(df_test['isFraud'], bin_pred_cal, rownames=['Actual'], colnames=['Pred_0.5'])
print(ct_bin)

"""
Script to run the full feature ablation experiments on XGBoost
for Phase 4.1 Independent Model Validation.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, precision_score, recall_score

from src.models.xgboost_model import FraudXGBoostClassifier
from src.evaluation.metrics import calculate_precision_recall_at_k, calculate_classification_metrics


def get_feature_groups(all_features: List[str]) -> Dict[str, List[str]]:
    balance_features = [
        "oldbalanceOrg",
        "oldbalanceDest",
        "balance_orig_drain_ratio",
        "is_orig_zero_balance",
        "is_dest_zero_balance",
        "orig_balance_coverage",
        "dest_balance_amount_ratio",
    ]
    velocity_features = [
        "orig_tx_count_1h",
        "orig_tx_count_6h",
        "orig_tx_count_24h",
        "orig_tx_count_72h",
        "orig_amount_sum_24h",
        "orig_amount_mean_24h",
        "orig_amount_std_24h",
        "dest_inflow_count_24h",
        "dest_inflow_sum_24h",
        "orig_amount_deviation_24h",
        "orig_amount_zscore_24h",
        "is_first_interaction",
    ]
    type_features = [
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    ]
    amount_features = [
        "amount",
        "amount_log",
    ]
    temporal_features = [
        "hour_of_day",
        "hour_sin",
        "hour_cos",
        "day_of_month",
        "is_merchant_dest",
    ]

    return {
        "balance": [f for f in balance_features if f in all_features],
        "velocity": [f for f in velocity_features if f in all_features],
        "type": [f for f in type_features if f in all_features],
        "amount": [f for f in amount_features if f in all_features],
        "temporal": [f for f in temporal_features if f in all_features],
    }


def define_experiments(all_features: List[str]) -> Dict[str, Tuple[str, List[str]]]:
    groups = get_feature_groups(all_features)
    bal = groups["balance"]
    vel = groups["velocity"]
    typ = groups["type"]
    amt = groups["amount"]
    tmp = groups["temporal"]

    experiments = {
        "Exp_A_Full_Model": (
            "Full feature model (all features)",
            all_features,
        ),
        "Exp_B_No_Balance": (
            "Remove all balance-derived features",
            [f for f in all_features if f not in bal],
        ),
        "Exp_C_No_Velocity": (
            "Remove all historical velocity features",
            [f for f in all_features if f not in vel],
        ),
        "Exp_D_No_Type": (
            "Remove transaction-type indicators",
            [f for f in all_features if f not in typ],
        ),
        "Exp_E_No_Amount": (
            "Remove transaction amount and amount-derived features",
            [f for f in all_features if f not in amt],
        ),
        "Exp_F_Balance_Only": (
            "Balance-only features",
            bal,
        ),
        "Exp_G_Type_Only": (
            "Transaction-type indicators only",
            typ,
        ),
        "Exp_H1_Velocity_Only": (
            "Historical velocity features only",
            vel,
        ),
        "Exp_H2_Amount_Only": (
            "Amount features only",
            amt,
        ),
        "Exp_H3_Balance_Plus_Type": (
            "Balance + Transaction Type features",
            bal + typ,
        ),
        "Exp_H4_Balance_Plus_Amount": (
            "Balance + Amount features",
            bal + amt,
        ),
    }

    return experiments


def main():
    print("=" * 80)
    print("PHASE 4.1: FEATURE ABLATION & LEAKAGE PROXY INVESTIGATION")
    print("=" * 80)

    # 1. Load Parquet Data
    print("\nLoading dataset partitions ...")
    df_train = pd.read_parquet("data/processed/train.parquet")
    df_val = pd.read_parquet("data/processed/val.parquet")
    df_test = pd.read_parquet("data/processed/test.parquet")

    y_train = df_train["isFraud"].values
    y_val = df_val["isFraud"].values
    y_test = df_test["isFraud"].values

    all_features = [
        c for c in df_train.columns
        if c not in ["isFraud", "isFlaggedFraud", "step", "partition", "nameOrig", "nameDest", "type"]
    ]
    print(f"Total features identified: {len(all_features)}")

    experiments = define_experiments(all_features)
    print(f"Defined {len(experiments)} ablation experiments.")

    # We evaluate Precision@K and Recall@K at K = 1,281 on test set (250 alerts/day)
    k_eval_test = 1281
    k_eval_val = 1250

    results = []

    for exp_id, (desc, feats) in experiments.items():
        print(f"\n--- Running {exp_id} ({len(feats)} features): {desc} ---")
        t0 = time.time()

        X_tr = df_train[feats].fillna(0.0)
        X_va = df_val[feats].fillna(0.0)
        X_te = df_test[feats].fillna(0.0)

        # Initialize standard XGBoost
        model = FraudXGBoostClassifier(
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=5.0,
            subsample=0.80,
            colsample_bytree=0.80,
            n_estimators=300,
            early_stopping_rounds=30,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_tr, y_train, X_va, y_val, feature_names=feats)
        fit_time = time.time() - t0

        # Predict probabilities
        p_val = model.predict_proba(X_va)[:, 1]
        p_test = model.predict_proba(X_te)[:, 1]

        # Validation Metrics
        val_pr_auc = float(average_precision_score(y_val, p_val))
        val_roc_auc = float(roc_auc_score(y_val, p_val))
        val_brier = float(brier_score_loss(y_val, p_val))

        # Test Metrics
        test_pr_auc = float(average_precision_score(y_test, p_test))
        test_roc_auc = float(roc_auc_score(y_test, p_test))
        test_brier = float(brier_score_loss(y_test, p_test))

        # Binary metrics at 0.5 threshold on test
        y_pred_test = (p_test >= 0.5).astype(int)
        test_prec = float(precision_score(y_test, y_pred_test, zero_division=0))
        test_rec = float(recall_score(y_test, y_pred_test, zero_division=0))

        # Precision@K and Recall@K on test (K=1281)
        pk_rk_test = calculate_precision_recall_at_k(y_test, p_test, [k_eval_test])
        prec_at_k = pk_rk_test[f"k_{k_eval_test}"]["precision_at_k"]
        rec_at_k = pk_rk_test[f"k_{k_eval_test}"]["recall_at_k"]

        print(f"  Fit time: {fit_time:.1f}s | Val PR-AUC: {val_pr_auc:.4f} | Test PR-AUC: {test_pr_auc:.4f} | Test ROC-AUC: {test_roc_auc:.4f}")
        print(f"  Test Prec: {test_prec:.4f} | Rec: {test_rec:.4f} | Prec@{k_eval_test}: {prec_at_k:.2%} | Rec@{k_eval_test}: {rec_at_k:.2%} | Brier: {test_brier:.6f}")

        results.append({
            "experiment_id": exp_id,
            "description": desc,
            "feature_count": len(feats),
            "val_pr_auc": val_pr_auc,
            "test_pr_auc": test_pr_auc,
            "test_roc_auc": test_roc_auc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_prec_at_k": prec_at_k,
            "test_rec_at_k": rec_at_k,
            "test_brier_score": test_brier,
            "fit_time_seconds": round(fit_time, 2),
            "features_used": ";".join(feats),
        })

    # Save to CSV
    res_df = pd.DataFrame(results)
    out_csv = Path("reports/feature_ablation_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved feature ablation results to: {out_csv}")

    # Print summary table
    print("\n" + "=" * 100)
    print("ABLATION SUMMARY TABLE")
    print("=" * 100)
    cols_display = ["experiment_id", "feature_count", "val_pr_auc", "test_pr_auc", "test_roc_auc", "test_prec_at_k", "test_rec_at_k", "test_brier_score"]
    print(res_df[cols_display].to_string(index=False))


if __name__ == "__main__":
    main()


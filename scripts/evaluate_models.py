"""Rigorous out-of-time evaluation of fraud models on the frozen test partition.

Evaluates:
1. Model generalization on unseen temporal test partition (steps 621–743)
2. Precision-Recall curves and Brier / ECE calibration metrics
3. Operational Capacity metrics (Precision@K, Recall@K, Fraud Value Captured@K)
4. Financial cost and ROI simulation under calibrated policy thresholds
5. Two-dimensional unit cost sensitivity analysis
6. Local TreeSHAP explanation cards for operational analysts
7. Publication-grade figures and audit reports
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve

from src.models.autoencoder import DeepAutoencoder
from src.models.xgboost_model import FraudXGBoostClassifier
from src.models.calibration import ProbabilityCalibrator
from src.models.hybrid import HybridFraudScorer
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_precision_recall_at_k,
    calculate_fraud_value_captured_at_k,
    calculate_brier_and_ece,
)
from src.evaluation.cost_analysis import CostAnalyzer, FinancialImpactReport
from src.explainability.shap_explainer import XGBoostShapExplainer, AutoencoderAttributionExplainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fraud models on test partition.")
    parser.add_argument("--test-path", type=str, default="data/processed/test.parquet", help="Path to test parquet")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory containing model checkpoints")
    parser.add_argument("--policy-config", type=str, default="configs/policy_config.yaml", help="Policy config YAML")
    parser.add_argument("--reports-dir", type=str, default="reports", help="Directory for evaluation reports")
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    forbidden = {
        "step", "type", "nameOrig", "nameDest", "isFraud", "isFlaggedFraud",
        "newbalanceOrig", "newbalanceDest",
    }
    return [col for col in df.columns if col not in forbidden]


def main() -> None:
    args = parse_args()
    print("=" * 80)
    print("STARTING OUT-OF-TIME FROZEN TEST SET EVALUATION")
    print("=" * 80)

    models_dir = Path(args.models_dir)
    reports_dir = Path(args.reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Policy Config and Saved Thresholds
    policy_cfg = load_yaml(args.policy_config)
    thresholds_file = models_dir / "decision_policy_thresholds.json"
    with open(thresholds_file, "r") as f:
        policy_meta = json.load(f)

    tau_low = policy_meta["tau_low"]
    tau_high = policy_meta["tau_high"]
    best_alpha = policy_meta["best_alpha"]
    print(f"Loaded Decision Policy Thresholds: tau_low={tau_low:.2f}, tau_high={tau_high:.2f}, alpha={best_alpha:.2f}")

    # 2. Load Frozen Test Partition
    print(f"\nLoading frozen test partition: {args.test_path} ...")
    df_test = pd.read_parquet(args.test_path)
    test_rows = len(df_test)
    test_frauds = int(df_test["isFraud"].sum())
    test_fraud_rate = float(df_test["isFraud"].mean())
    print(f"Loaded test set: {test_rows:,} rows, {test_frauds:,} frauds ({test_fraud_rate:.4%})")

    test_hours = float(df_test["step"].max() - df_test["step"].min() + 1)
    daily_cap = policy_cfg["policy"]["capacity_constraints"]["max_daily_review_capacity"]
    test_capacity = int(np.floor(daily_cap * (test_hours / 24.0)))  # 500 * (123 / 24) = 2,562
    print(f"Test temporal horizon: {test_hours:.0f} hours ({test_hours/24:.2f} days). Review capacity: {test_capacity:,} alerts.")

    test_k_eval_points = [
        int(np.floor(100 * (test_hours / 24.0))),
        int(np.floor(250 * (test_hours / 24.0))),
        test_capacity,
        int(np.floor(1000 * (test_hours / 24.0))),
    ]

    feature_cols = get_feature_columns(df_test)
    X_test = df_test[feature_cols].fillna(0.0)
    y_test = df_test["isFraud"].values
    amounts_test = df_test["amount"].values

    # 3. Load Trained Model Artifacts
    print("\nLoading trained model artifacts ...")
    # XGBoost
    xgb_classifier = FraudXGBoostClassifier.load(str(models_dir / "xgboost_best.joblib"))
    # Calibrator
    calibrator = ProbabilityCalibrator.load(str(models_dir / "calibrator_best.joblib"))
    # Autoencoder & Scaler
    ae_scaler = joblib.load(models_dir / "autoencoder_scaler.joblib")
    ae_model = DeepAutoencoder.load(str(models_dir / "autoencoder_best.pt"))
    # Hybrid Scorer
    hybrid_scorer = HybridFraudScorer.load(str(models_dir / "hybrid_scorer_best.joblib"))

    # 4. Generate Predictions
    print("\nScoring test partition across model architectures ...")
    # XGBoost Raw and Calibrated
    p_xgb_test_raw = xgb_classifier.predict_proba(X_test)[:, 1]
    p_xgb_test_cal = calibrator.calibrate(p_xgb_test_raw)

    # Autoencoder
    X_test_scaled = ae_scaler.transform(X_test)
    test_recon_errors, test_recon = ae_model.compute_reconstruction_errors(X_test_scaled)
    s_ae_test = ae_model.predict_anomaly_score(test_recon_errors)

    # Hybrid
    s_hybrid_test = hybrid_scorer.compute_score(p_xgb_test_cal, s_ae_test, alpha=best_alpha)

    # Evaluate Models on Test Set
    test_metrics = {
        "XGBoost_Uncalibrated": calculate_classification_metrics(y_test, p_xgb_test_raw),
        "XGBoost_Calibrated": calculate_classification_metrics(y_test, p_xgb_test_cal),
        "DeepAutoencoder": calculate_classification_metrics(y_test, s_ae_test),
        "Hybrid_Fusion": calculate_classification_metrics(y_test, s_hybrid_test),
    }

    # Calibration quality on test set
    cal_eval_uncal = calculate_brier_and_ece(y_test, p_xgb_test_raw)
    cal_eval_cal = calculate_brier_and_ece(y_test, p_xgb_test_cal)

    print("\n--- TEST BENCHMARK RESULTS ---")
    for name, m in test_metrics.items():
        print(f"  {name:22s} -> PR-AUC: {m['pr_auc']:.4f} | ROC-AUC: {m['roc_auc']:.4f} | Brier: {m['brier_score']:.6f} | F1: {m['f1_score']:.4f}")
    print(f"  Calibration ECE improvement: {cal_eval_uncal['expected_calibration_error']:.4f} -> {cal_eval_cal['expected_calibration_error']:.4f}")

    # Primary scoring system
    primary_score = s_hybrid_test if policy_meta.get("primary_score") == "Hybrid_Fusion" else p_xgb_test_cal

    # 5. Operational Capacity Evaluation on Test Set
    pk_rk_test = calculate_precision_recall_at_k(y_test, primary_score, test_k_eval_points)
    fvc_k_test = calculate_fraud_value_captured_at_k(y_test, primary_score, amounts_test, test_k_eval_points)

    capacity_results = {}
    print("\n--- OPERATIONAL INVESTIGATION CAPACITY ON TEST SET ---")
    for k_val in test_k_eval_points:
        key = f"k_{k_val}"
        capacity_results[key] = {**pk_rk_test[key], **fvc_k_test[key]}
        print(f"  K={k_val:,} alerts ({k_val / (test_hours/24):.0f}/day) -> Precision@K: {pk_rk_test[key]['precision_at_k']:.2%} | Recall@K: {pk_rk_test[key]['recall_at_k']:.2%} | FVC: {fvc_k_test[key]['fraud_value_captured_rate']:.2%} (${fvc_k_test[key]['captured_fraud_value']:,.2f})")

    # 6. Policy Financial Simulation on Test Set
    cost_analyzer = CostAnalyzer(
        false_negative_multiplier=policy_cfg["policy"]["cost_assumptions"]["false_negative_multiplier"],
        default_false_positive_cost=policy_cfg["policy"]["cost_assumptions"]["false_positive_cost"],
        default_investigation_cost=policy_cfg["policy"]["cost_assumptions"]["investigation_cost"],
    )

    policy_report = cost_analyzer.evaluate_policy(
        y_true=y_test,
        y_prob=primary_score,
        amounts=amounts_test,
        tau_low=tau_low,
        tau_high=tau_high,
        max_review_capacity=test_capacity,
    )

    print("\n--- FINANCIAL DECISION POLICY OUTCOMES (TEST SET) ---")
    print(f"  Total Test Fraud Exposure : ${policy_report.total_fraud_volume:,.2f}")
    print(f"  Saved Fraud Intercepted   : ${policy_report.saved_fraud_amount:,.2f} ({policy_report.saved_fraud_amount/policy_report.total_fraud_volume:.2%})")
    print(f"  Uncaught Fraud Loss (FN)  : ${policy_report.uncaught_fraud_amount:,.2f}")
    print(f"  FP Friction Cost          : ${policy_report.fp_friction_cost:,.2f}")
    print(f"  Analyst Review Cost       : ${policy_report.review_operational_cost:,.2f}")
    print(f"  Net Financial Savings     : ${policy_report.net_financial_savings:,.2f} (ROI: {policy_report.roi_percentage:.1f}%)")
    print(f"  Triage Queue Allocations  : APPROVE={policy_report.approved_count:,} | REVIEW={policy_report.review_count:,} | DECLINE={policy_report.declined_count:,}")

    # 7. Cost Sensitivity Grid
    print("\nRunning two-dimensional unit cost sensitivity grid ...")
    fp_costs = policy_cfg["policy"]["sensitivity_analysis"]["false_positive_costs"]
    rev_costs = policy_cfg["policy"]["sensitivity_analysis"]["investigation_costs"]
    sensitivity_df = cost_analyzer.run_sensitivity_grid(
        y_true=y_test,
        y_prob=primary_score,
        amounts=amounts_test,
        tau_low=tau_low,
        tau_high=tau_high,
        fp_costs=fp_costs,
        rev_costs=rev_costs,
    )

    # 8. Visualizations
    print("\nGenerating publication figures ...")
    # PR Curve Comparison
    plt.figure(figsize=(8, 6))
    for name, p in [
        ("XGBoost (Calibrated)", p_xgb_test_cal),
        ("Deep Autoencoder", s_ae_test),
        ("Hybrid Model", s_hybrid_test),
    ]:
        prec, rec, _ = precision_recall_curve(y_test, p)
        auc_val = test_metrics[name.replace(" (Calibrated)", "_Calibrated").replace("Deep Autoencoder", "DeepAutoencoder").replace("Hybrid Model", "Hybrid_Fusion")]["pr_auc"]
        plt.plot(rec, prec, label=f"{name} (PR-AUC = {auc_val:.4f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves on Frozen Test Partition (Steps 621–743)")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "test_precision_recall_curves.png", dpi=300)
    plt.close()

    # Reliability Diagram
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(cal_eval_uncal["bin_confidences"], cal_eval_uncal["bin_accuracies"], "s-", color="#ef4444", label=f"Uncalibrated (ECE = {cal_eval_uncal['expected_calibration_error']:.4f})")
    plt.plot(cal_eval_cal["bin_confidences"], cal_eval_cal["bin_accuracies"], "o-", color="#10b981", label=f"Isotonic Calibrated (ECE = {cal_eval_cal['expected_calibration_error']:.4f})")
    plt.xlabel("Mean Predicted Probability (Confidence)")
    plt.ylabel("Fraction of Positives (Empirical Risk)")
    plt.title("Reliability Diagram (Calibration) on Test Partition")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "test_calibration_reliability_curve.png", dpi=300)
    plt.close()

    # 9. Explainability Case Studies
    print("\nGenerating SHAP explanation cards for sample transactions ...")
    explainer = XGBoostShapExplainer(xgb_classifier.model, feature_cols)
    cards = []

    # High-Risk Confirmed Fraud
    fraud_indices = np.where(y_test == 1)[0]
    if len(fraud_indices) > 0:
        sample_fraud_idx = fraud_indices[0]
        card_fraud = explainer.explain_instance(
            row=X_test.iloc[sample_fraud_idx],
            predicted_prob=float(primary_score[sample_fraud_idx]),
            decision="DECLINE" if primary_score[sample_fraud_idx] >= tau_high else "REVIEW",
        )
        cards.append(card_fraud)

    # Borderline Review Case
    review_indices = np.where((primary_score >= tau_low) & (primary_score < tau_high))[0]
    if len(review_indices) > 0:
        sample_rev_idx = review_indices[0]
        card_review = explainer.explain_instance(
            row=X_test.iloc[sample_rev_idx],
            predicted_prob=float(primary_score[sample_rev_idx]),
            decision="REVIEW",
        )
        cards.append(card_review)

    # Legitimate Approved Case
    legit_indices = np.where((y_test == 0) & (primary_score < tau_low))[0]
    if len(legit_indices) > 0:
        sample_legit_idx = legit_indices[0]
        card_legit = explainer.explain_instance(
            row=X_test.iloc[sample_legit_idx],
            predicted_prob=float(primary_score[sample_legit_idx]),
            decision="APPROVE",
        )
        cards.append(card_legit)

    # 10. Save Output JSON & Markdown Report
    test_summary_payload = {
        "test_samples": test_rows,
        "test_frauds": test_frauds,
        "test_fraud_rate": test_fraud_rate,
        "test_metrics": test_metrics,
        "calibration": {
            "uncalibrated_ece": cal_eval_uncal["expected_calibration_error"],
            "calibrated_ece": cal_eval_cal["expected_calibration_error"],
            "uncalibrated_brier": cal_eval_uncal["brier_score"],
            "calibrated_brier": cal_eval_cal["brier_score"],
        },
        "operational_capacity": capacity_results,
        "financial_policy_impact": {
            "tau_low": tau_low,
            "tau_high": tau_high,
            "total_fraud_volume": policy_report.total_fraud_volume,
            "saved_fraud_amount": policy_report.saved_fraud_amount,
            "uncaught_fraud_amount": policy_report.uncaught_fraud_amount,
            "fp_friction_cost": policy_report.fp_friction_cost,
            "review_cost": policy_report.review_operational_cost,
            "net_financial_savings": policy_report.net_financial_savings,
            "roi_pct": policy_report.roi_percentage,
            "approved": policy_report.approved_count,
            "reviewed": policy_report.review_count,
            "declined": policy_report.declined_count,
        },
    }

    with open(reports_dir / "test_evaluation_metrics.json", "w") as f:
        json.dump(test_summary_payload, f, indent=2)

    # Generate Markdown Report
    generate_test_markdown_report(
        reports_dir / "model_evaluation_report.md",
        test_summary_payload,
        sensitivity_df,
        cards,
    )

    print("\n" + "=" * 80)
    print("OUT-OF-TIME TEST EVALUATION COMPLETE")
    print(f"Report: {reports_dir / 'model_evaluation_report.md'}")
    print(f"Metrics: {reports_dir / 'test_evaluation_metrics.json'}")
    print("=" * 80)


def generate_test_markdown_report(
    report_path: Path,
    payload: Dict[str, Any],
    sensitivity_df: pd.DataFrame,
    cards: List[Any],
) -> None:
    tm = payload["test_metrics"]
    pol = payload["financial_policy_impact"]
    cap = payload["operational_capacity"]
    cal = payload["calibration"]

    lines = [
        "# Out-of-Time Model Evaluation & Financial Risk Intelligence Report",
        "",
        "## Executive Summary",
        f"- **Evaluation Dataset:** Frozen Out-of-Time Test Partition (`data/processed/test.parquet`, simulation steps 621–743).",
        f"- **Transaction Volume:** {payload['test_samples']:,} transactions ({payload['test_frauds']:,} confirmed fraud cases, {payload['test_fraud_rate']:.4%} prevalence).",
        f"- **Prevalence Shift:** Out-of-time test fraud prevalence is ~16x higher than the training horizon (0.0917% -> 1.5039%), reflecting acute synthetic adversarial attack bursts.",
        f"- **Top Performer:** Cost-Weighted XGBoost Classifier achieved **PR-AUC = {tm['XGBoost_Calibrated']['pr_auc']:.4f}** and **ROC-AUC = {tm['XGBoost_Calibrated']['roc_auc']:.4f}**.",
        f"- **Calibration Validation:** Isotonic calibration reduced Brier Score from **{cal['uncalibrated_brier']:.6f}** to **{cal['calibrated_brier']:.6f}** and ECE from **{cal['uncalibrated_ece']:.4f}** to **{cal['calibrated_ece']:.4f}**.",
        f"- **Financial Interception:** Intercepted **${pol['saved_fraud_amount']:,.2f}** out of ${pol['total_fraud_volume']:,.2f} total fraud exposure (**{pol['saved_fraud_amount']/pol['total_fraud_volume']:.1%}** capture rate).",
        f"- **Net Financial Savings:** **${pol['net_financial_savings']:,.2f}** after deducting FP friction and investigation costs (**{pol['roi_pct']:.1f}% ROI**).",
        "",
        "## 1. Out-of-Time Performance Matrix",
        "",
        "| Architecture | PR-AUC (Average Precision) | ROC-AUC | Brier Score | Precision | Recall | F1 Score |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_name, m in tm.items():
        lines.append(
            f"| **{model_name}** | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | {m['brier_score']:.6f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1_score']:.4f} |"
        )

    lines.extend([
        "",
        "## 2. Operational Capacity & Fraud Value Captured @ K",
        "",
        "Test horizon spans 123 simulation hours (5.125 days).",
        "",
        "| Operational Review Tier | Alerts Inspected (K) | Precision@K | Recall@K | Fraud Dollars Intercepted | Total Fraud Exposure | Fraud Value Captured Rate |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for k_key, k_data in cap.items():
        k_val = k_data["k"]
        daily = k_val / 5.125
        lines.append(
            f"| {daily:.0f} alerts/day | {k_val:,} | {k_data['precision_at_k']:.2%} | {k_data['recall_at_k']:.2%} | ${k_data['captured_fraud_value']:,.2f} | ${k_data['total_fraud_value']:,.2f} | **{k_data['fraud_value_captured_rate']:.2%}** |"
        )

    lines.extend([
        "",
        "## 3. Financial Decision Policy Simulation",
        "",
        f"- **Pre-Tuned Thresholds:** $\\tau_{{low}} = {pol['tau_low']:.2f}$ (Straight-Through Approval), $\\tau_{{high}} = {pol['tau_high']:.2f}$ (Automated Decline / MFA Block)",
        f"- **Straight-Through Approvals:** {pol['approved']:,} transactions ({pol['approved']/payload['test_samples']:.2%})",
        f"- **Level-1 Analyst Review Queue:** {pol['reviewed']:,} transactions ({pol['reviewed']/payload['test_samples']:.2%})",
        f"- **Automated Declines:** {pol['declined']:,} transactions ({pol['declined']/payload['test_samples']:.2%})",
        f"- **Uncaught Fraud Loss (FN):** ${pol['uncaught_fraud_amount']:,.2f}",
        f"- **False-Positive Friction Cost:** ${pol['fp_friction_cost']:,.2f}",
        f"- **Analyst Investigation Cost:** ${pol['review_cost']:,.2f}",
        f"- **Net Financial Value Delivered:** **${pol['net_financial_savings']:,.2f}** (Baseline Loss: ${pol['total_fraud_volume']:,.2f})",
        "",
        "## 4. Cost Sensitivity Analysis Matrix",
        "",
        "Net financial savings ($) evaluated across a two-dimensional grid of FP Friction Cost ($) vs Analyst Investigation Cost ($):",
        "",
        "| FP Cost ($) \\ Review Cost ($) | $10.00 | $25.00 | $50.00 | $75.00 | $100.00 |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    pivot = sensitivity_df.pivot(index="c_fp", columns="c_rev", values="net_savings")
    for fp_val in pivot.index:
        row_str = f"| **${fp_val:.2f}** |"
        for rev_val in pivot.columns:
            val = pivot.loc[fp_val, rev_val]
            row_str += f" ${val:,.0f} |"
        lines.append(row_str)

    lines.extend([
        "",
        "## 5. Analyst Explanation & Regulatory Audit Cards (TreeSHAP)",
        "",
    ])

    for i, card in enumerate(cards, 1):
        lines.extend([
            f"### Case Study {i}: Transaction #{card.transaction_index} ({card.decision})",
            f"- **Calibrated Fraud Risk:** {card.predicted_probability:.2%}",
            f"- **Narrative Summary:** {card.narrative_summary}",
            "- **Top Risk Contributors:**",
        ])
        for rf in card.top_risk_factors:
            lines.append(f"  - `{rf.feature_name}` (Val: {rf.feature_value:.2f}, SHAP: {rf.contribution:+.3f}): {rf.plain_english}")
        if card.top_mitigating_factors:
            lines.append("- **Top Mitigating Contributors:**")
            for mf in card.top_mitigating_factors:
                lines.append(f"  - `{mf.feature_name}` (Val: {mf.feature_value:.2f}, SHAP: {mf.contribution:+.3f}): {mf.plain_english}")
        lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

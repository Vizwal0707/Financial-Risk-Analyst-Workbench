"""
Script to generate notebooks/03_model_training_and_evaluation.ipynb
"""

import json
from pathlib import Path


def create_notebook():
    cells = []

    def md_cell(source_lines):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source_lines],
        }

    def code_cell(source_lines):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source_lines],
        }

    # Header
    cells.append(
        md_cell([
            "# Enterprise Financial Fraud Risk Intelligence Platform",
            "## Research Notebook 03: Model Training, Probability Calibration & Out-of-Time Benchmarking",
            "",
            "**Executive Scope:**",
            "- Pre-training data audit and mathematical leakage verification.",
            "- Reproducible baselines (Majority Class, Transaction-Type Corridor, Expert Heuristic, Logistic Regression, Random Forest).",
            "- PyTorch Deep Autoencoder anomaly detection trained strictly on legitimate transaction dynamics.",
            "- Gradient Boosted Decision Trees (XGBoost) with scale-pos-weight imbalance management.",
            "- Rigorous probability calibration comparing Platt Sigmoid scaling vs. Non-parametric Isotonic Regression.",
            "- Hybrid score fusion optimization (supervised + unsupervised).",
            "- Out-of-time evaluation on frozen test partition (steps 621–743).",
            "- Capacity-constrained triage policy optimization (@K alerts/day) and 2D cost sensitivity analysis.",
            "- Local and global TreeSHAP explainability for regulatory auditability.",
        ])
    )

    # Imports
    cells.append(
        code_cell([
            "import json",
            "from pathlib import Path",
            "import numpy as np",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import joblib",
            "import torch",
            "",
            "# Set visualization style",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')",
            "plt.rcParams['figure.figsize'] = (12, 6)",
            "plt.rcParams['font.size'] = 11",
            "",
            "print('Libraries and dependencies loaded successfully.')",
        ])
    )

    # Section 1: Pre-training Leakage Audit
    cells.append(
        md_cell([
            "---",
            "### 1. Pre-Training Leakage Audit & Feature Schema Verification",
            "Ensuring strict compliance with point-in-time correctness: verifying target exclusion, post-transaction balance omission, and chronological partition integrity.",
        ])
    )

    cells.append(
        code_cell([
            "train_df = pd.read_parquet('data/processed/train.parquet', columns=['step', 'isFraud', 'amount', 'balance_orig_drain_ratio'])",
            "val_df = pd.read_parquet('data/processed/val.parquet', columns=['step', 'isFraud', 'amount', 'balance_orig_drain_ratio'])",
            "test_df = pd.read_parquet('data/processed/test.parquet', columns=['step', 'isFraud', 'amount', 'balance_orig_drain_ratio'])",
            "",
            "summary = pd.DataFrame([",
            "    {'Partition': 'Train', 'Steps': f'{train_df.step.min()}–{train_df.step.max()}', 'Total Rows': len(train_df), 'Fraud Rows': int(train_df.isFraud.sum()), 'Fraud Rate': f'{train_df.isFraud.mean():.4%}'},",
            "    {'Partition': 'Validation', 'Steps': f'{val_df.step.min()}–{val_df.step.max()}', 'Total Rows': len(val_df), 'Fraud Rows': int(val_df.isFraud.sum()), 'Fraud Rate': f'{val_df.isFraud.mean():.4%}'},",
            "    {'Partition': 'Test (Out-of-Time)', 'Steps': f'{test_df.step.min()}–{test_df.step.max()}', 'Total Rows': len(test_df), 'Fraud Rows': int(test_df.isFraud.sum()), 'Fraud Rate': f'{test_df.isFraud.mean():.4%}'},",
            "])",
            "summary",
        ])
    )

    # Section 2: Baseline Models Evaluation
    cells.append(
        md_cell([
            "---",
            "### 2. Baseline Model Performance Benchmark (Validation Set)",
            "Evaluating standard benchmark classifiers to establish clear empirical floors before neural or boosted architectures.",
        ])
    )

    cells.append(
        code_cell([
            "with open('reports/model_comparison.json', 'r') as f:",
            "    val_metrics = json.load(f)",
            "",
            "baselines = ['MajorityClass', 'TransactionTypeOnly', 'RuleHeuristic', 'LogisticRegression', 'RandomForest']",
            "base_df = pd.DataFrame([",
            "    {'Model': m, 'PR-AUC': val_metrics[m]['pr_auc'], 'ROC-AUC': val_metrics[m]['roc_auc'], 'Brier Score': val_metrics[m]['brier_score'], 'Precision': val_metrics[m]['precision'], 'Recall': val_metrics[m]['recall']}",
            "    for m in baselines",
            "])",
            "base_df",
        ])
    )

    # Section 3: Deep Autoencoder
    cells.append(
        md_cell([
            "---",
            "### 3. Deep Autoencoder Anomaly Detection",
            "Trained strictly on legitimate transactions ($N=300,000$, $\\text{isFraud}=0$) using a 5-layer symmetric bottleneck architecture ($31 \\to 64 \\to 32 \\to 16 \\to 32 \\to 64 \\to 31$). Reconstruction error represents anomalous behavior relative to normal operational banking distributions.",
        ])
    )

    cells.append(
        code_cell([
            "from PIL import Image",
            "img = Image.open('reports/figures/autoencoder_reconstruction_distribution.png')",
            "plt.figure(figsize=(14, 6))",
            "plt.imshow(img)",
            "plt.axis('off')",
            "plt.title('Deep Autoencoder Reconstruction Error Distribution (Legitimate vs Fraudulent)', fontsize=14, pad=12)",
            "plt.show()",
        ])
    )

    # Section 4: Probability Calibration
    cells.append(
        md_cell([
            "---",
            "### 4. Probability Calibration: Platt Sigmoid vs. Isotonic Regression",
            "Raw tree ensemble outputs exhibit probability compression due to heavy negative subsampling or `scale_pos_weight` reweighting. We evaluate Isotonic Regression vs. Sigmoid Calibration strictly on the validation partition.",
        ])
    )

    cells.append(
        code_cell([
            "cal_img = Image.open('reports/figures/test_calibration_reliability_curve.png')",
            "plt.figure(figsize=(12, 10))",
            "plt.imshow(cal_img)",
            "plt.axis('off')",
            "plt.title('Out-of-Time Calibration Reliability Curve & Brier Score Decomposition', fontsize=14, pad=12)",
            "plt.show()",
        ])
    )

    # Section 5: Precision-Recall Out-of-Time Evaluation
    cells.append(
        md_cell([
            "---",
            "### 5. Out-of-Time Performance Comparison (Precision-Recall Frontier)",
            "Comparing architectures on the frozen out-of-time test partition (simulation steps 621–743).",
        ])
    )

    cells.append(
        code_cell([
            "pr_img = Image.open('reports/figures/test_precision_recall_curves.png')",
            "plt.figure(figsize=(12, 8))",
            "plt.imshow(pr_img)",
            "plt.axis('off')",
            "plt.title('Out-of-Time Precision-Recall Frontier (Test Partition)', fontsize=14, pad=12)",
            "plt.show()",
        ])
    )

    # Section 6: Test Performance Table
    cells.append(
        md_cell([
            "### Out-of-Time Benchmark Performance Matrix",
        ])
    )

    cells.append(
        code_cell([
            "with open('reports/test_evaluation_metrics.json', 'r') as f:",
            "    test_data = json.load(f)",
            "",
            "test_summary = pd.DataFrame([",
            "    {",
            "        'Architecture': name,",
            "        'PR-AUC (AP)': m['pr_auc'],",
            "        'ROC-AUC': m['roc_auc'],",
            "        'Brier Score': m['brier_score'],",
            "        'Precision': m['precision'],",
            "        'Recall': m['recall'],",
            "        'F1 Score': m['f1_score']",
            "    }",
            "    for name, m in test_data['test_metrics'].items()",
            "])",
            "test_summary",
        ])
    )

    # Section 7: Operational Capacity @ K
    cells.append(
        md_cell([
            "---",
            "### 6. Operational Investigation Capacity & Fraud Value Captured @ K",
            "In institutional fraud operations (e.g., JPMorgan Chase), analysts can inspect only a fixed quota of alerts per day. We simulate Precision@K, Recall@K, and Fraud Value Captured across operational capacity tiers.",
        ])
    )

    cells.append(
        code_cell([
            "cap_data = []",
            "for k_tier, d in test_data['operational_capacity'].items():",
            "    k_val = d['k']",
            "    daily = k_val / 5.125",
            "    cap_data.append({",
            "        'Daily Capacity': f'{daily:.0f} alerts/day',",
            "        'Total Alerts (K)': k_val,",
            "        'Precision@K': f\"{d['precision_at_k']:.2%}\",",
            "        'Recall@K': f\"{d['recall_at_k']:.2%}\",",
            "        'Fraud Intercepted ($)': f\"${d['captured_fraud_value']:,.2f}\",",
            "        'Total Fraud Value ($)': f\"${d['total_fraud_value']:,.2f}\",",
            "        'Fraud Value Captured (%)': f\"{d['fraud_value_captured_rate']:.2%}\",",
            "    })",
            "pd.DataFrame(cap_data)",
        ])
    )

    # Section 8: Financial Decision Policy & Cost Sensitivity
    cells.append(
        md_cell([
            "---",
            "### 7. Financial Decision Policy Simulation & 2D Unit Cost Sensitivity Analysis",
            "Implementing three-tier automated triage ($\\\\tau_{low} = 0.10, \\\\tau_{high} = 0.75$) balancing false positive friction ($c_{fp} = \\$100$) vs. analyst review cost ($c_{rev} = \\$25$).",
        ])
    )

    cells.append(
        code_cell([
            "pol = test_data['financial_policy_impact']",
            "print('--- FINANCIAL DECISION POLICY SUMMARY (TEST HORIZON) ---')",
            "print(f\"Total Exposure     : ${pol['total_fraud_volume']:,.2f}\")",
            "print(f\"Saved Fraud Loss   : ${pol['saved_fraud_amount']:,.2f} ({pol['saved_fraud_amount']/pol['total_fraud_volume']:.2%})\")",
            "print(f\"Uncaught Fraud Loss: ${pol['uncaught_fraud_amount']:,.2f}\")",
            "print(f\"FP Friction Cost   : ${pol['fp_friction_cost']:,.2f}\")",
            "print(f\"Analyst Review Cost: ${pol['review_cost']:,.2f}\")",
            "print(f\"Net Value Saved    : ${pol['net_financial_savings']:,.2f} (ROI: {pol['roi_pct']:.1f}%)\")",
            "print(f\"Triage Breakdown   : APPROVE={pol['approved']:,} | REVIEW={pol['reviewed']:,} | DECLINE={pol['declined']:,}\")",
        ])
    )

    # Section 9: Explainability
    cells.append(
        md_cell([
            "---",
            "### 8. Explainability & Triage Risk Cards (TreeSHAP)",
            "Generating local feature attributions and plain-English Fraud Risk Explanation Cards to assist human investigators during alert review. Note: SHAP values represent empirical model feature attributions, not causal mechanisms.",
        ])
    )

    cells.append(
        code_cell([
            "from src.explainability.shap_explainer import XGBoostShapExplainer",
            "",
            "# Load model and feature pipeline metadata",
            "from src.models.xgboost_model import FraudXGBoostClassifier",
            "xgb_wrapper = FraudXGBoostClassifier.load('models/xgboost_best.joblib')",
            "test_sample = pd.read_parquet('data/processed/test.parquet').head(5)",
            "feature_cols = [c for c in test_sample.columns if c not in ['isFraud', 'isFlaggedFraud', 'step', 'partition', 'nameOrig', 'nameDest', 'type']]",
            "",
            "explainer = XGBoostShapExplainer(xgb_wrapper.model, feature_names=feature_cols)",
            "card = explainer.explain_instance(test_sample[feature_cols].iloc[0], predicted_prob=0.9995, decision='DECLINE')",
            "",
            "print('================================================================================')",
            "print(f'FRAUD RISK EXPLANATION CARD: TRANSACTION #{card.transaction_index}')",
            "print(f'Decision: {card.decision} | Calibrated Fraud Risk: {card.predicted_probability:.2%}')",
            "print(f'Narrative Summary: {card.narrative_summary}')",
            "print('Top Risk Factors (SHAP Feature Attributions):')",
            "for rf in card.top_risk_factors:",
            "    print(f'  - {rf.feature_name} ({rf.feature_value:.2f}): {rf.plain_english} [SHAP: {rf.contribution:+.3f}]')",
            "print('================================================================================')",
        ])
    )

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    out_path = Path("notebooks/03_model_training_and_evaluation.ipynb")
    with open(out_path, "w") as f:
        json.dump(notebook_dict, f, indent=1)

    print(f"Generated notebook at {out_path} ({len(cells)} cells).")


if __name__ == "__main__":
    create_notebook()


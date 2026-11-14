"""Unit test suite for Phase 4 models, calibration, ensembling, and evaluation."""

import numpy as np
import pandas as pd
import pytest
from typing import Any, Dict, List, Tuple

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
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_precision_recall_at_k,
    calculate_fraud_value_captured_at_k,
    calculate_brier_and_ece,
)
from src.evaluation.cost_analysis import CostAnalyzer
from src.evaluation.threshold_optimizer import DecisionPolicyOptimizer
from src.explainability.shap_explainer import XGBoostShapExplainer, AutoencoderAttributionExplainer


@pytest.fixture
def synthetic_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generates a small deterministic tabular financial dataset."""
    np.random.seed(42)
    n = 200
    features = {
        "amount": np.random.uniform(10, 500000, n),
        "amount_log": np.random.uniform(2, 13, n),
        "oldbalanceOrg": np.random.uniform(0, 1000000, n),
        "oldbalanceDest": np.random.uniform(0, 1000000, n),
        "balance_orig_drain_ratio": np.random.uniform(0, 1, n),
        "is_orig_zero_balance": np.random.choice([0, 1], n, p=[0.8, 0.2]),
        "is_dest_zero_balance": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "type_TRANSFER": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "type_CASH_OUT": np.random.choice([0, 1], n, p=[0.6, 0.4]),
        "orig_tx_count_1h": np.random.poisson(1.0, n),
    }
    df = pd.DataFrame(features)
    y = np.zeros(n, dtype=np.int64)
    # Fraud condition: Transfer + high drain
    fraud_mask = (df["type_TRANSFER"] == 1) & (df["balance_orig_drain_ratio"] > 0.8)
    y[fraud_mask] = 1
    amounts = df["amount"].values
    return df, y, amounts


def test_baseline_majority_class(synthetic_data):
    X, y, _ = synthetic_data
    clf = MajorityClassClassifier().fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    preds = clf.predict(X)
    assert np.all(preds == 0)


def test_baseline_transaction_type(synthetic_data):
    X, y, _ = synthetic_data
    clf = TransactionTypeClassifier().fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all((probs[:, 1] >= 0.0) & (probs[:, 1] <= 1.0))


def test_baseline_rule_heuristic(synthetic_data):
    X, y, _ = synthetic_data
    clf = RuleHeuristicClassifier(amount_threshold=100000.0)
    probs = clf.predict_proba(X)
    assert probs.shape == (len(X), 2)
    preds = clf.predict(X)
    assert set(preds).issubset({0, 1})


def test_baseline_logistic_regression(synthetic_data):
    X, y, _ = synthetic_data
    clf = LogisticRegressionBaseline(max_iter=100).fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all((probs[:, 1] >= 0.0) & (probs[:, 1] <= 1.0))


def test_deep_autoencoder_architecture_and_training(synthetic_data):
    X, y, _ = synthetic_data
    # Filter strictly legitimate rows
    legit_mask = (y == 0)
    X_legit = X[legit_mask].values.astype(np.float32)

    ae = DeepAutoencoder(
        input_dim=X.shape[1],
        encoder_hidden_dims=[16, 8],
        decoder_hidden_dims=[8, 16],
        device="cpu",
        random_seed=42,
    )
    trainer = AutoencoderTrainer(
        model=ae,
        epochs=3,
        batch_size=32,
        learning_rate=0.01,
        patience=2,
    )
    res = trainer.fit(X_legit, X_legit)
    assert res["epochs_trained"] >= 1
    assert "best_val_loss" in res

    errors, recon = ae.compute_reconstruction_errors(X.values)
    assert len(errors) == len(X)
    assert np.all(errors >= 0.0)

    # Percentile calibration mapping
    scores = ae.predict_anomaly_score(errors)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_xgboost_classifier(synthetic_data):
    X, y, _ = synthetic_data
    model = FraudXGBoostClassifier(
        n_estimators=10,
        max_depth=3,
        tree_method="hist",
        random_state=42,
    )
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all((probs[:, 1] >= 0.0) & (probs[:, 1] <= 1.0))

    importances = model.get_feature_importances()
    assert len(importances) == X.shape[1]
    assert "importance_gain" in importances.columns


def test_probability_calibrator_sigmoid_and_isotonic(synthetic_data):
    X, y, _ = synthetic_data
    raw_probs = np.random.uniform(0.01, 0.99, len(y))

    # Sigmoid
    sig_cal = ProbabilityCalibrator(method="sigmoid").fit(raw_probs, y)
    p_sig = sig_cal.calibrate(raw_probs)
    assert np.all((p_sig >= 0.0) & (p_sig <= 1.0))

    # Isotonic
    iso_cal = ProbabilityCalibrator(method="isotonic").fit(raw_probs, y)
    p_iso = iso_cal.calibrate(raw_probs)
    assert np.all((p_iso >= 0.0) & (p_iso <= 1.0))

    # Method comparison
    comp = ProbabilityCalibrator.compare_methods(raw_probs, y)
    assert "uncalibrated" in comp
    assert "sigmoid" in comp
    assert "isotonic" in comp


def test_hybrid_fraud_scorer(synthetic_data):
    _, y, _ = synthetic_data
    n = len(y)
    p_xgb = np.random.uniform(0.0, 1.0, n)
    s_ae = np.random.uniform(0.0, 1.0, n)

    scorer = HybridFraudScorer(alpha=0.7)
    s_hyb = scorer.compute_score(p_xgb, s_ae)
    assert np.all((s_hyb >= 0.0) & (s_hyb <= 1.0))

    opt = scorer.optimize_alpha(p_xgb, s_ae, y, alpha_grid=[0.0, 0.5, 1.0])
    assert 0.0 <= opt["best_alpha"] <= 1.0
    assert "xgb_standalone_pr_auc" in opt


def test_evaluation_metrics_and_fvc(synthetic_data):
    _, y, amounts = synthetic_data
    probs = np.random.uniform(0.0, 1.0, len(y))

    metrics = calculate_classification_metrics(y, probs, threshold=0.5)
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "brier_score" in metrics

    pk_rk = calculate_precision_recall_at_k(y, probs, [10, 25])
    assert "k_10" in pk_rk
    assert "precision_at_k" in pk_rk["k_10"]

    fvc = calculate_fraud_value_captured_at_k(y, probs, amounts, [10, 25])
    assert "k_10" in fvc
    assert "fraud_value_captured_rate" in fvc["k_10"]

    cal = calculate_brier_and_ece(y, probs, n_bins=5)
    assert "expected_calibration_error" in cal
    assert "maximum_calibration_error" in cal


def test_cost_analyzer_and_threshold_optimizer(synthetic_data):
    _, y, amounts = synthetic_data
    probs = np.random.uniform(0.0, 1.0, len(y))

    analyzer = CostAnalyzer()
    rep = analyzer.evaluate_policy(
        y_true=y,
        y_prob=probs,
        amounts=amounts,
        tau_low=0.3,
        tau_high=0.7,
    )
    assert rep.total_financial_loss >= 0.0
    assert rep.approved_count + rep.review_count + rep.declined_count == len(y)

    opt = DecisionPolicyOptimizer(cost_analyzer=analyzer)
    thresholds = opt.optimize_thresholds(
        y_true=y,
        y_prob=probs,
        amounts=amounts,
        max_capacity=50,
    )
    assert thresholds.tau_low <= thresholds.tau_high
    assert thresholds.review_volume <= 50


def test_explainability_engines(synthetic_data):
    X, y, _ = synthetic_data
    xgb = FraudXGBoostClassifier(n_estimators=5, max_depth=2, random_state=42).fit(X, y)

    explainer = XGBoostShapExplainer(xgb.model, list(X.columns))
    card = explainer.explain_instance(X.iloc[0], predicted_prob=0.85, decision="DECLINE")
    assert card.predicted_probability == 0.85
    assert len(card.narrative_summary) > 0

    ae_explainer = AutoencoderAttributionExplainer(list(X.columns))
    orig = X.iloc[0].values
    recon = orig * 0.9
    attribs = ae_explainer.explain_anomaly(orig, recon, anomaly_score=0.9)
    assert len(attribs) <= 5
    assert "percentage_contribution" in attribs[0]


def test_batch_vs_single_inference_parity(synthetic_data):
    """Verify that batch prediction and single-record inference yield identical outputs."""
    X, y, _ = synthetic_data
    xgb = FraudXGBoostClassifier(n_estimators=10, max_depth=3, random_state=42).fit(X, y)
    cal = ProbabilityCalibrator(method="isotonic").fit(xgb.predict_proba(X)[:, 1], y)

    # Batch prediction
    p_batch_raw = xgb.predict_proba(X)[:, 1]
    p_batch_cal = cal.calibrate(p_batch_raw)

    # Single-record predictions
    p_single_raw = []
    p_single_cal = []
    for i in range(len(X)):
        row_df = X.iloc[[i]]
        raw_p = xgb.predict_proba(row_df)[0, 1]
        cal_p = cal.calibrate(np.array([raw_p]))[0]
        p_single_raw.append(raw_p)
        p_single_cal.append(cal_p)

    np.testing.assert_allclose(p_batch_raw, np.array(p_single_raw), atol=1e-6)
    np.testing.assert_allclose(p_batch_cal, np.array(p_single_cal), atol=1e-6)


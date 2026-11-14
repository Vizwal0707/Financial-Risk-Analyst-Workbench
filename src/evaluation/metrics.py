"""Rigorous financial fraud evaluation metrics.

Covers:
- Precision-Recall AUC (Average Precision) and ROC-AUC
- Brier Score and Expected Calibration Error (ECE)
- Operational Precision@K and Recall@K
- Financial Fraud Value Captured (FVC@K)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def calculate_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate comprehensive classification metrics at given threshold."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    y_pred_arr = (y_prob_arr >= threshold).astype(np.int64)

    # Check class presence
    n_pos = np.sum(y_true_arr == 1)
    n_neg = np.sum(y_true_arr == 0)

    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr)) if n_pos > 0 else 0.0
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr)) if (n_pos > 0 and n_neg > 0) else 0.5
    brier = float(brier_score_loss(y_true_arr, y_prob_arr))

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_samples": int(len(y_true_arr)),
        "total_fraud_samples": int(n_pos),
    }


def calculate_precision_recall_at_k(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    k_list: List[int],
) -> Dict[str, Dict[str, float]]:
    """Calculate Precision@K and Recall@K where K is the number of highest-risk alerts investigated."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)

    total_positives = float(np.sum(y_true_arr == 1))
    sort_indices = np.argsort(-y_prob_arr)
    sorted_labels = y_true_arr[sort_indices]

    results: Dict[str, Dict[str, float]] = {}
    n = len(y_true_arr)

    for k in k_list:
        actual_k = min(k, n)
        if actual_k <= 0:
            continue
        top_k_labels = sorted_labels[:actual_k]
        tp_k = float(np.sum(top_k_labels == 1))

        prec_k = tp_k / actual_k
        rec_k = (tp_k / total_positives) if total_positives > 0 else 0.0

        results[f"k_{k}"] = {
            "k": actual_k,
            "true_positives_at_k": int(tp_k),
            "precision_at_k": float(prec_k),
            "recall_at_k": float(rec_k),
            "alert_rate": float(actual_k / n),
        }

    return results


def calculate_fraud_value_captured_at_k(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    amounts: Union[pd.Series, np.ndarray],
    k_list: List[int],
) -> Dict[str, Dict[str, float]]:
    """Calculate financial fraud dollars intercepted vs total fraud exposure at operational capacity K."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float64)
    amounts_arr = np.asarray(amounts, dtype=np.float64)

    fraud_mask = (y_true_arr == 1)
    total_fraud_value = float(np.sum(amounts_arr[fraud_mask]))

    sort_indices = np.argsort(-y_prob_arr)
    sorted_labels = y_true_arr[sort_indices]
    sorted_amounts = amounts_arr[sort_indices]

    results: Dict[str, Dict[str, float]] = {}
    n = len(y_true_arr)

    for k in k_list:
        actual_k = min(k, n)
        if actual_k <= 0:
            continue

        top_k_labels = sorted_labels[:actual_k]
        top_k_amounts = sorted_amounts[:actual_k]

        captured_fraud_value = float(np.sum(top_k_amounts[top_k_labels == 1]))
        fvc_rate = (captured_fraud_value / total_fraud_value) if total_fraud_value > 0 else 0.0

        results[f"k_{k}"] = {
            "k": actual_k,
            "captured_fraud_value": captured_fraud_value,
            "total_fraud_value": total_fraud_value,
            "fraud_value_captured_rate": float(fvc_rate),
            "unintercepted_fraud_value": float(total_fraud_value - captured_fraud_value),
        }

    return results


def calculate_brier_and_ece(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Dict[str, Any]:
    """Calculate calibration metrics: Brier score, Expected Calibration Error (ECE), and Maximum Calibration Error (MCE)."""
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)

    brier = float(np.mean((y_prob_arr - y_true_arr) ** 2))

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(y_prob_arr, quantiles * 100)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.linspace(0, 1, n_bins + 1)
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    mce = 0.0
    n = len(y_true_arr)

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            in_bin = (y_prob_arr >= lower) & (y_prob_arr <= upper)
        else:
            in_bin = (y_prob_arr >= lower) & (y_prob_arr < upper)

        count = int(np.sum(in_bin))
        if count > 0:
            acc = float(np.mean(y_true_arr[in_bin]))
            conf = float(np.mean(y_prob_arr[in_bin]))
            diff = abs(acc - conf)
            ece += (count / n) * diff
            mce = max(mce, diff)
            bin_accuracies.append(acc)
            bin_confidences.append(conf)
            bin_counts.append(count)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(float((lower + upper) / 2))
            bin_counts.append(0)

    return {
        "brier_score": brier,
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges.tolist(),
    }


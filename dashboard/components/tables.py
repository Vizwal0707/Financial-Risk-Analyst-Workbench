"""Table rendering components for the risk workbench."""

from typing import List, Optional
import pandas as pd
import streamlit as st
from dashboard.config.theme import COLORS
from dashboard.utils.formatters import format_currency, format_latency, format_percentage, format_step_time, mask_account


def render_investigation_table(df: pd.DataFrame, max_rows: int = 50) -> None:
    """Renders formatted queue table for investigation and audit."""
    if df.empty:
        st.info("No transactions match the selected filter criteria.")
        return

    view_df = df.head(max_rows).copy()

    # Formatter dict for display
    display_cols = [
        "request_id",
        "step",
        "corridor_type",
        "amount",
        "name_orig_masked",
        "name_dest_masked",
        "calibrated_probability",
        "decision",
        "action_code",
        "risk_weighted_investigation_priority",
    ]

    available_cols = [c for c in display_cols if c in view_df.columns]
    render_df = view_df[available_cols].copy()

    # Rename columns for presentation
    rename_map = {
        "request_id": "Request ID",
        "step": "Step",
        "corridor_type": "Mechanism",
        "amount": "Amount ($)",
        "name_orig_masked": "Origin Account",
        "name_dest_masked": "Destination Account",
        "calibrated_probability": "Calibrated Prob",
        "decision": "Risk Decision",
        "action_code": "Action Code",
        "risk_weighted_investigation_priority": "Priority Score",
    }
    render_df = render_df.rename(columns=rename_map)

    if "Amount ($)" in render_df.columns:
        render_df["Amount ($)"] = render_df["Amount ($)"].apply(lambda x: f"${x:,.2f}")
    if "Calibrated Prob" in render_df.columns:
        render_df["Calibrated Prob"] = render_df["Calibrated Prob"].apply(lambda x: f"{x:.4f}")
    if "Priority Score" in render_df.columns:
        render_df["Priority Score"] = render_df["Priority Score"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "—")

    st.dataframe(render_df, use_container_width=True, hide_index=True)


def render_ablation_table(df: pd.DataFrame) -> None:
    """Renders Phase 4.1 feature ablation comparison table."""
    if df.empty:
        st.warning("Feature ablation results unavailable.")
        return

    display_df = df[[
        "experiment_id",
        "description",
        "feature_count",
        "val_pr_auc",
        "test_pr_auc",
        "test_roc_auc",
        "test_prec_at_k",
        "test_rec_at_k",
    ]].copy()

    display_df.columns = [
        "Experiment",
        "Description",
        "Features",
        "Val PR-AUC",
        "Test PR-AUC",
        "Test ROC-AUC",
        "Prec@1281",
        "Recall@1281",
    ]

    for col in ["Val PR-AUC", "Test PR-AUC", "Test ROC-AUC", "Prec@1281", "Recall@1281"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)


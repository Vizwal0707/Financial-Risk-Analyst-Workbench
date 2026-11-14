"""Plotly charts for the institutional risk workbench."""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.config.theme import COLORS, PLOTLY_TEMPLATE


def plot_decision_donut(
    approve_cnt: int,
    review_cnt: int,
    decline_cnt: int,
    title: str = "Operational Decision Distribution",
) -> go.Figure:
    """Generates an institutional donut chart for decision volumes."""
    labels = ["APPROVE (STP)", "REVIEW (Analyst Queue)", "DECLINE (Block/Step-up)"]
    values = [approve_cnt, review_cnt, decline_cnt]
    colors = [COLORS["approve"], COLORS["review"], COLORS["decline"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color=COLORS["bg_card"], width=2)),
                textinfo="label+percent",
                hoverinfo="label+value+percent",
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title),
        showlegend=False,
        height=320,
    )
    return fig


def plot_corridor_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart showing transaction count and fraud rate by corridor mechanism."""
    if df.empty or "corridor_type" not in df.columns:
        return go.Figure()

    summary = df.groupby("corridor_type").agg(
        total_tx=("amount", "count"),
        total_volume=("amount", "sum"),
        mean_prob=("calibrated_probability", "mean"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=summary["corridor_type"],
            y=summary["total_tx"],
            name="Transaction Count",
            marker_color=COLORS["accent_blue"],
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary["corridor_type"],
            y=summary["mean_prob"],
            name="Mean Calibrated Fraud Prob",
            marker_color=COLORS["decline"],
            mode="lines+markers",
            yaxis="y2",
            line=dict(width=2, dash="dot"),
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Volume & Average Fraud Risk by Payment Mechanism"),
        xaxis=dict(title="Transaction Mechanism"),
        yaxis=dict(title="Transaction Count", showgrid=True),
        yaxis2=dict(
            title="Mean Fraud Probability",
            overlaying="y",
            side="right",
            range=[0, max(0.1, summary["mean_prob"].max() * 1.3)],
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=340,
    )
    return fig


def plot_step_timeline(df: pd.DataFrame) -> go.Figure:
    """Step-by-step temporal volume trend with risk level."""
    if df.empty or "step" not in df.columns:
        return go.Figure()

    step_grp = df.groupby("step").agg(
        tx_count=("amount", "count"),
        declines=("decision", lambda s: (s == "DECLINE").sum()),
        reviews=("decision", lambda s: (s == "REVIEW").sum()),
    ).reset_index().sort_values("step")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=step_grp["step"],
            y=step_grp["tx_count"],
            name="Total Scored",
            line=dict(color=COLORS["accent_cyan"], width=2),
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=step_grp["step"],
            y=step_grp["declines"],
            name="Declines",
            line=dict(color=COLORS["decline"], width=2),
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=step_grp["step"],
            y=step_grp["reviews"],
            name="Reviews",
            line=dict(color=COLORS["review"], width=1.5, dash="dash"),
            mode="lines",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Transaction Scoring Volume by Simulation Step"),
        xaxis=dict(title="Simulation Step (1 Step = 1 Hour)"),
        yaxis=dict(title="Scored Transactions"),
        height=340,
    )
    return fig


def plot_confusion_heatmap(cm_data: Dict[str, int]) -> go.Figure:
    """Renders 3x2 / 3x3 operational confusion matrix heatmap."""
    z = [
        [cm_data.get("legit_approve", 0), cm_data.get("legit_review", 0), cm_data.get("legit_decline", 0)],
        [cm_data.get("fraud_approve", 0), cm_data.get("fraud_review", 0), cm_data.get("fraud_decline", 0)],
    ]
    x = ["APPROVE (STP)", "REVIEW (Queue)", "DECLINE (Block)"]
    y = ["Actual Legitimate", "Actual Fraud"]

    annotations = []
    for i, row in enumerate(z):
        for j, val in enumerate(row):
            annotations.append(
                dict(
                    x=x[j],
                    y=y[i],
                    text=f"{val:,}",
                    font=dict(color=COLORS["text_main"], size=13, family="monospace"),
                    showarrow=False,
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0, COLORS["bg_card"]], [1, COLORS["accent_blue"]]],
            showscale=False,
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="Operational 3x2 Confusion Matrix"),
        annotations=annotations,
        height=260,
    )
    return fig


def plot_shap_horizontal(top_risk: List[Dict[str, Any]], top_mitigating: List[Dict[str, Any]]) -> go.Figure:
    """Renders dual-sided SHAP feature contribution bar chart."""
    features = []
    values = []
    colors = []

    # Risk-inducing features (positive)
    for item in reversed(top_risk[:6]):
        features.append(item.get("feature_name", ""))
        values.append(float(item.get("contribution", 0.0)))
        colors.append(COLORS["decline"])

    # Risk-mitigating features (negative)
    for item in top_mitigating[:6]:
        features.append(item.get("feature_name", ""))
        values.append(float(item.get("contribution", 0.0)))
        colors.append(COLORS["approve"])

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="TreeSHAP Feature Attributions (Log-Odds Impact)"),
        xaxis=dict(title="Contribution to Fraud Log-Odds (Positive = Risk, Negative = Mitigating)"),
        yaxis=dict(title="Feature"),
        height=360,
    )
    return fig


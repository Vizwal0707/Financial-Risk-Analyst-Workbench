"""Module E: Interactive Policy Simulator & Cost Sensitivity Optimizer."""

import streamlit as st
import pandas as pd
from dashboard.config.theme import COLORS, get_institutional_css
from dashboard.components.cards import render_kpi_card, render_disclaimer_banner
from dashboard.components.charts import plot_decision_donut, plot_confusion_heatmap
from dashboard.components.sidebar import render_sidebar
from dashboard.services.simulation_service import simulation_service
from dashboard.utils.formatters import format_currency, format_percentage

st.set_page_config(
    page_title="Policy Simulator | Fraud Risk Intelligence",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(get_institutional_css(), unsafe_allow_html=True)
render_sidebar()

# Page Header
st.markdown(
    f"""
    <div style="margin-bottom: 20px;">
        <h1 style="font-size: 1.85rem; font-weight: 700; color: {COLORS['text_main']}; margin: 0;">
            Interactive Policy Simulator & Financial Cost Optimizer
        </h1>
        <p style="font-size: 0.88rem; color: {COLORS['text_muted']}; margin: 4px 0 0 0;">
            Hypothetical sensitivity analysis over pre-scored offline evaluation data. Strictly separated between validation optimization and frozen test benchmark.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
render_disclaimer_banner(
    "OFFLINE SIMULATION ONLY. Threshold adjustments in this simulator operate on cached evaluation data. "
    "Production API thresholds (&tau;low=0.10, &tau;high=0.75) remain frozen and unmodified."
)

# Partition Selection for Rigorous Model Governance
dataset_option = st.radio(
    "Evaluation Partition:",
    [
        "Validation Partition (Steps 501–620, N=209,984) [Recommended for Threshold Exploration]",
        "Frozen Out-of-Time Test Partition (Steps 621–743, N=90,829) [Audited Baseline Benchmark]",
    ],
    index=0,
    horizontal=True,
    help="Model governance requirement: explore thresholds on validation partition to prevent test set data leakage."
)

if "Validation Partition" in dataset_option:
    target_data_path = "data/processed/val_evaluation_scored.parquet"
    partition_label = "Validation Partition (Steps 501–620, N=209,984)"
else:
    target_data_path = "data/processed/test_evaluation_scored.parquet"
    partition_label = "Frozen Out-of-Time Test Partition (Steps 621–743, N=90,829)"

sim_data = pd.read_parquet(target_data_path)

# Sidebar / Panel Parameters for Simulation
st.markdown(f"### Hypothetical Decision Boundary & Financial Parameters — *({partition_label})*")
param_col1, param_col2, param_col3 = st.columns(3)

with param_col1:
    sim_tau_low = st.slider("Low-Risk Threshold (tau_low):", min_value=0.01, max_value=0.40, value=0.10, step=0.01)
    sim_tau_high = st.slider("High-Risk Threshold (tau_high):", min_value=0.45, max_value=0.99, value=0.75, step=0.01)

with param_col2:
    sim_c_fp = st.number_input("Hypothetical False Positive Cost ($):", min_value=1.0, max_value=1000.0, value=100.0, step=10.0)
    sim_c_rev = st.number_input("Hypothetical L1 Review Cost ($):", min_value=1.0, max_value=500.0, value=25.0, step=5.0)

with param_col3:
    sim_daily_cap = st.slider("Investigator Daily Capacity (cases/day):", min_value=50, max_value=2000, value=500, step=50)
    sim_fn_mult = st.slider("Fraud Loss Exposure Multiplier:", min_value=1.0, max_value=3.0, value=1.0, step=0.1)

# Run Simulation
sim_results = simulation_service.simulate(
    tau_low=sim_tau_low,
    tau_high=sim_tau_high,
    c_fp=sim_c_fp,
    c_rev=sim_c_rev,
    daily_capacity=sim_daily_cap,
    fn_multiplier=sim_fn_mult,
    df=sim_data,
)

if "error" in sim_results:
    st.error(sim_results["error"])
    st.stop()

# Baseline Comparison (Audited Phase 4.1 Policy on Selected Partition)
baseline_results = simulation_service.simulate(
    tau_low=0.10,
    tau_high=0.75,
    c_fp=100.0,
    c_rev=25.0,
    daily_capacity=500,
    fn_multiplier=1.0,
    df=sim_data,
)

st.markdown("<hr style='border: none; border-top: 1px solid #3A506B; margin: 15px 0 20px 0;'>", unsafe_allow_html=True)

# Primary Simulation Metrics
vol = sim_results["volumes"]
cm = sim_results["confusion_matrix"]
fin = sim_results["financial_impact"]
op = sim_results["operational"]

k1, k2, k3, k4 = st.columns(4)
with k1:
    diff_savings = fin["net_savings"] - baseline_results["financial_impact"]["net_savings"]
    render_kpi_card(
        "Net Financial Savings",
        format_currency(fin["net_savings"]),
        subtitle=f"Diff from Baseline: {format_currency(diff_savings)}",
        source="Intercepted Fraud - Operational Costs - Missed Fraud",
        border_color=COLORS["approve"] if fin["net_savings"] > 0 else COLORS["decline"],
    )
with k2:
    render_kpi_card(
        "Operational Feasibility",
        f"{op['daily_review_rate']:.1f} cases / day",
        subtitle=f"Capacity: {sim_daily_cap} | {'EXCEEDED' if op['capacity_exceeded'] else 'FEASIBLE'}",
        source="5-Day Horizon Evaluation Window",
        border_color=COLORS["decline"] if op["capacity_exceeded"] else COLORS["accent_cyan"],
    )
with k3:
    render_kpi_card(
        "Simulated Policy Errors",
        f"{cm['legit_decline']} FP | {cm['fraud_approve']} FN",
        subtitle=f"Precision: {format_percentage(op['precision'])} | Recall: {format_percentage(op['recall'])}",
        source="Hypothetical 3x2 Confusion Split",
        border_color=COLORS["review"],
    )
with k4:
    render_kpi_card(
        "Simulated Review Load",
        f"{vol['review']:,} ({format_percentage(vol['review_pct'], 2)})",
        subtitle=f"Approved: {vol['approve']:,} | Declined: {vol['decline']:,}",
        source="Operational Volume Allocation",
        border_color=COLORS["accent_blue"],
    )

st.markdown("<br>", unsafe_allow_html=True)

# Visual Comparison Row
c_col1, c_col2 = st.columns([1, 1])
with c_col1:
    fig_donut = plot_decision_donut(
        approve_cnt=vol["approve"],
        review_cnt=vol["review"],
        decline_cnt=vol["decline"],
        title="Simulated Policy Decision Distribution",
    )
    st.plotly_chart(fig_donut, use_container_width=True)

with c_col2:
    fig_cm = plot_confusion_heatmap(cm)
    st.plotly_chart(fig_cm, use_container_width=True)

# Comparative Table: Audited Baseline vs. Simulated Policy
st.markdown("### Audited Baseline Policy vs. Simulated Alternative")

comp_df = pd.DataFrame([
    {
        "Policy Dimension": "Low-Risk Threshold (tau_low)",
        "Audited Baseline Policy": "0.10",
        "Simulated Alternative Policy": f"{sim_tau_low:.2f}",
        "Delta / Impact": f"{sim_tau_low - 0.10:+.2f}",
    },
    {
        "Policy Dimension": "High-Risk Threshold (tau_high)",
        "Audited Baseline Policy": "0.75",
        "Simulated Alternative Policy": f"{sim_tau_high:.2f}",
        "Delta / Impact": f"{sim_tau_high - 0.75:+.2f}",
    },
    {
        "Policy Dimension": "Straight-Through Approvals (STP)",
        "Audited Baseline Policy": f"{baseline_results['volumes']['approve']:,} ({format_percentage(baseline_results['volumes']['approve_pct'])})",
        "Simulated Alternative Policy": f"{vol['approve']:,} ({format_percentage(vol['approve_pct'])})",
        "Delta / Impact": f"{vol['approve'] - baseline_results['volumes']['approve']:+,} txns",
    },
    {
        "Policy Dimension": "Manual Review Volume (L1 Queue)",
        "Audited Baseline Policy": f"{baseline_results['volumes']['review']:,} ({format_percentage(baseline_results['volumes']['review_pct'], 3)})",
        "Simulated Alternative Policy": f"{vol['review']:,} ({format_percentage(vol['review_pct'], 3)})",
        "Delta / Impact": f"{vol['review'] - baseline_results['volumes']['review']:+,} txns",
    },
    {
        "Policy Dimension": "Immediate Declines / Blocks",
        "Audited Baseline Policy": f"{baseline_results['volumes']['decline']:,} ({format_percentage(baseline_results['volumes']['decline_pct'])})",
        "Simulated Alternative Policy": f"{vol['decline']:,} ({format_percentage(vol['decline_pct'])})",
        "Delta / Impact": f"{vol['decline'] - baseline_results['volumes']['decline']:+,} txns",
    },
    {
        "Policy Dimension": "False Positives (Legitimate Declined)",
        "Audited Baseline Policy": f"{baseline_results['confusion_matrix']['legit_decline']} txns",
        "Simulated Alternative Policy": f"{cm['legit_decline']} txns",
        "Delta / Impact": f"{cm['legit_decline'] - baseline_results['confusion_matrix']['legit_decline']:+,} txns",
    },
    {
        "Policy Dimension": "Missed Frauds (Undetected Approved)",
        "Audited Baseline Policy": f"{baseline_results['confusion_matrix']['fraud_approve']} txns",
        "Simulated Alternative Policy": f"{cm['fraud_approve']} txns",
        "Delta / Impact": f"{cm['fraud_approve'] - baseline_results['confusion_matrix']['fraud_approve']:+,} txns",
    },
    {
        "Policy Dimension": "Net Financial Savings ($)",
        "Audited Baseline Policy": format_currency(baseline_results['financial_impact']['net_savings']),
        "Simulated Alternative Policy": format_currency(fin['net_savings']),
        "Delta / Impact": format_currency(diff_savings),
    },
])

st.dataframe(comp_df, use_container_width=True, hide_index=True)


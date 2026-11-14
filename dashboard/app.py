"""Main entrance for the Institutional Financial Fraud Risk Workbench & Governance Dashboard."""

import streamlit as st
from dashboard.config.settings import settings
from dashboard.config.theme import COLORS, get_institutional_css
from dashboard.components.cards import render_kpi_card, render_disclaimer_banner
from dashboard.components.sidebar import render_sidebar
from dashboard.services.api_client import api_client
from dashboard.services.audit_service import audit_service
from dashboard.utils.formatters import format_currency

st.set_page_config(
    page_title=f"{settings.app_title}",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply Institutional CSS Styling
st.markdown(get_institutional_css(), unsafe_allow_html=True)

# Render Persistent Global Sidebar
render_sidebar()

# Main Title & Hero Banner
st.markdown(
    f"""
    <div style="padding: 10px 0 20px 0;">
        <div style="font-size: 0.85rem; font-weight: 700; color: {COLORS['accent_cyan']}; letter-spacing: 0.1em; text-transform: uppercase;">
            JPMorgan Chase & Co. | Financial Crime Risk Management & Quantitative Analytics
        </div>
        <h1 style="font-size: 2.2rem; font-weight: 800; color: {COLORS['text_main']}; margin: 4px 0 8px 0; letter-spacing: -0.02em;">
            Financial Fraud Detection & Risk Intelligence Platform
        </h1>
        <p style="font-size: 0.95rem; color: {COLORS['text_muted']}; max-width: 900px; line-height: 1.5; margin: 0;">
            Institutional transaction monitoring, cost-sensitive machine learning decisioning, TreeSHAP explainability, 
            and model governance console evaluated on the synthetic PaySim benchmark (6,362,620 transactions).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_disclaimer_banner()

# Quick Portfolio Status Ribbon
audit_summary = audit_service.get_audit_summary()
ready_status = api_client.check_ready()

q_col1, q_col2, q_col3, q_col4 = st.columns(4)
with q_col1:
    api_online = ready_status.get("status") == "ready"
    render_kpi_card(
        "FastAPI Scoring Engine",
        "ONLINE (Ready)" if api_online else "OFFLINE",
        subtitle="POST /v1/score (Inference SLA ≤ 50ms)",
        source="Liveness & Readiness Probes",
        border_color=COLORS["approve"] if api_online else COLORS["decline"],
    )
with q_col2:
    total_audited = audit_summary.get("total_count", 0)
    render_kpi_card(
        "Persisted Audit Logs",
        f"{total_audited:,} Scored",
        subtitle="SQLite 3.45 (Write-Ahead Logging)",
        source="audit_scoring_logs Table",
        border_color=COLORS["accent_cyan"],
    )
with q_col3:
    declined_val = audit_summary.get("declined_amount", 0.0)
    render_kpi_card(
        "Blocked Financial Value",
        format_currency(declined_val),
        subtitle=f"{audit_summary.get('decline_count', 0):,} High-Risk Transactions",
        source="Persistent Audit Database",
        border_color=COLORS["decline"],
    )
with q_col4:
    render_kpi_card(
        "Active Policy Baseline",
        "τ_low: 0.10 | τ_high: 0.75",
        subtitle="Cost-Sensitive Isotonic Calibrated",
        source="models/decision_policy_thresholds.json",
        border_color=COLORS["review"],
    )

if not api_online:
    st.warning(
        "⚡ **DEMO MODE: FastAPI Scoring Backend is currently offline.** "
        "Precomputed synthetic analytics, model governance benchmarks, and policy simulations remain fully interactive. "
        "Live transaction scoring (`POST /v1/score`) is disabled until the backend service is started."
    )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Institutional Risk Workbench Modules")
st.markdown("Select a module below or use the sidebar navigation to access specific operational and governance capabilities:")

m_col1, m_col2 = st.columns(2)

with m_col1:
    st.markdown(
        f"""
        <div class="risk-card">
            <h3 style="color: {COLORS['accent_cyan']}; margin-top: 0;">Module A: Executive Risk Overview</h3>
            <p style="color: {COLORS['text_muted']}; font-size: 0.88rem; line-height: 1.5;">
                Comprehensive portfolio KPIs, fraud exposure volume, operational decision split (STP Approvals, Reviews, Blocks), 
                and temporal transaction trends across hourly simulation steps.
            </p>
            <div style="font-size: 0.78rem; color: {COLORS['text_dim']};">Navigation: <code>01_Executive_Overview</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="risk-card">
            <h3 style="color: {COLORS['accent_blue']}; margin-top: 0;">Module B: Investigation Queue</h3>
            <p style="color: {COLORS['text_muted']}; font-size: 0.88rem; line-height: 1.5;">
                Priority-weighted analyst triage queue ranked monotonically by <code>P<sub>cal</sub> × ln(1 + amount)</code>. 
                Features multi-criteria filtering, account masking, and simulated case disposition tracking.
            </p>
            <div style="font-size: 0.78rem; color: {COLORS['text_dim']};">Navigation: <code>02_Investigation_Queue</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="risk-card">
            <h3 style="color: {COLORS['approve']}; margin-top: 0;">Module C: Transaction Explorer</h3>
            <p style="color: {COLORS['text_muted']}; font-size: 0.88rem; line-height: 1.5;">
                Interactive transaction scoring console connected live to the FastAPI backend. Generates instant 
                Fraud Risk Explanation Cards with local TreeSHAP attribution and reason-code justification.
            </p>
            <div style="font-size: 0.78rem; color: {COLORS['text_dim']};">Navigation: <code>03_Transaction_Explorer</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m_col2:
    st.markdown(
        f"""
        <div class="risk-card">
            <h3 style="color: {COLORS['review']}; margin-top: 0;">Module D: Model Governance & Lineage</h3>
            <p style="color: {COLORS['text_muted']}; font-size: 0.88rem; line-height: 1.5;">
                Regulatory validation benchmarking: Out-of-time test PR-AUC (0.9979), ROC-AUC (0.9996), Brier score calibration, 
                11 feature ablation experiments, and formal drift monitoring state.
            </p>
            <div style="font-size: 0.78rem; color: {COLORS['text_dim']};">Navigation: <code>04_Model_Governance</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="risk-card">
            <h3 style="color: {COLORS['decline']}; margin-top: 0;">Module E: Policy Simulator</h3>
            <p style="color: {COLORS['text_muted']}; font-size: 0.88rem; line-height: 1.5;">
                Interactive risk threshold and financial cost sensitivity simulator. Evaluate hypothetical operating points 
                on offline test data without modifying frozen production model thresholds.
            </p>
            <div style="font-size: 0.78rem; color: {COLORS['text_dim']};">Navigation: <code>05_Policy_Simulator</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="risk-card">
            <h3 style="color: {COLORS['accent_cyan']}; margin-top: 0;">Module F: Compliance Audit Explorer</h3>
            <p style="color: {COLORS['text_muted']}; font-size: 0.88rem; line-height: 1.5;">
                Persistent decision audit log inspector with PCI-DSS-aligned account masking best practice (non-certified prototype), 
                execution latency measurements, ACID durability verification, and full explanation JSON tracing.
            </p>
            <div style="font-size: 0.78rem; color: {COLORS['text_dim']};">Navigation: <code>06_Audit_Explorer</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


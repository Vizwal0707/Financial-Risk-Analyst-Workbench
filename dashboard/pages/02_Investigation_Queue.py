"""Module B: Investigation Queue & Priority Triage Workbench."""

import streamlit as st
import pandas as pd
from dashboard.config.theme import COLORS, get_institutional_css
from dashboard.components.cards import render_kpi_card, render_disclaimer_banner, get_risk_badge_html, render_reason_pills_html
from dashboard.components.tables import render_investigation_table
from dashboard.components.sidebar import render_sidebar
from dashboard.services.audit_service import audit_service
from dashboard.utils.formatters import format_currency, format_step_time, mask_account

st.set_page_config(
    page_title="Investigation Queue | Fraud Risk Intelligence",
    page_icon="⚖️",
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
            Investigation Queue & Priority Triage Workbench
        </h1>
        <p style="font-size: 0.88rem; color: {COLORS['text_muted']}; margin: 4px 0 0 0;">
            Risk-weighted triage queue ordered by Priority Score: <code>P<sub>cal</sub> × ln(1 + amount)</code>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
render_disclaimer_banner()

# Initialize Simulated Session Status Storage
if "analyst_case_statuses" not in st.session_state:
    st.session_state.analyst_case_statuses = {}

# Load Live Records from SQLite Audit Store
recent_df = audit_service.get_recent_logs(limit=2500)

if recent_df.empty:
    st.warning("Audit log is currently empty. Run transactions through the API or batch scorer to populate.")
    st.stop()

# Filter Controls
with st.expander("🔍 Filter & Search Queue", expanded=True):
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)
    with f_col1:
        selected_decision = st.selectbox("Decision Filter:", ["ALL", "REVIEW", "DECLINE", "APPROVE"], index=1)
    with f_col2:
        selected_tier = st.selectbox("Risk Tier Filter:", ["ALL", "HIGH", "MEDIUM", "LOW"], index=0)
    with f_col3:
        all_types = ["ALL"] + sorted(list(recent_df["corridor_type"].dropna().unique()))
        selected_type = st.selectbox("Mechanism Filter:", all_types, index=0)
    with f_col4:
        search_query = st.text_input("Search Request ID / Account:", placeholder="e.g. p51-bench or C1****7890")

# Apply Filters
filtered_df = recent_df.copy()
if selected_decision != "ALL":
    filtered_df = filtered_df[filtered_df["decision"] == selected_decision]
if selected_tier != "ALL":
    filtered_df = filtered_df[filtered_df["risk_tier"] == selected_tier]
if selected_type != "ALL":
    filtered_df = filtered_df[filtered_df["corridor_type"] == selected_type]
if search_query:
    q = search_query.strip().lower()
    filtered_df = filtered_df[
        filtered_df["request_id"].str.lower().str.contains(q, na=False) |
        filtered_df["name_orig_masked"].str.lower().str.contains(q, na=False) |
        filtered_df["name_dest_masked"].str.lower().str.contains(q, na=False)
    ]

# Sort by Priority Descending
filtered_df = filtered_df.sort_values("risk_weighted_investigation_priority", ascending=False)

# Metric Summary Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    render_kpi_card("Filtered Queue Size", f"{len(filtered_df):,} Cases", subtitle="Matching filter criteria", border_color=COLORS["accent_cyan"])
with kpi2:
    total_exposure = filtered_df["amount"].sum()
    render_kpi_card("Filtered Gross Exposure", format_currency(total_exposure), subtitle="Sum of transaction amounts", border_color=COLORS["decline"])
with kpi3:
    max_priority = filtered_df["risk_weighted_investigation_priority"].max() if not filtered_df.empty else 0.0
    render_kpi_card("Peak Priority Index", f"{max_priority:.2f}", subtitle="P_cal * ln(1 + amount)", border_color=COLORS["review"])
with kpi4:
    active_in_session = len(st.session_state.analyst_case_statuses)
    render_kpi_card("Simulated Analyst Actions", f"{active_in_session} Cases", subtitle="In-memory analyst session state", border_color=COLORS["approve"])

st.markdown("<br>", unsafe_allow_html=True)

# Main Queue Table
st.markdown("### Active Triage Queue (Top Ranked)")
render_investigation_table(filtered_df, max_rows=50)

# Transaction Detail & Simulated Analyst Action Console
st.markdown("<hr style='border: none; border-top: 1px solid #3A506B; margin: 25px 0 20px 0;'>", unsafe_allow_html=True)
st.markdown("### Analyst Action & Case Disposition Console")

sel_req_id = st.selectbox(
    "Select Transaction Request ID to Inspect / Dispose:",
    filtered_df["request_id"].tolist(),
    index=0 if not filtered_df.empty else None,
)

if sel_req_id:
    row = audit_service.get_record_by_request_id(sel_req_id)
    if row:
        d_col1, d_col2 = st.columns([2, 1])
        with d_col1:
            st.markdown(
                f"""
                <div class="risk-card" style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <span style="font-size: 1.1rem; font-weight: 700; color: {COLORS['text_main']};">
                            Request: <code>{row['request_id']}</code>
                        </span>
                        <span>{get_risk_badge_html(row['decision'])}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; font-size: 0.85rem; margin-bottom: 12px;">
                        <div><strong style="color: {COLORS['text_muted']};">Amount:</strong> {format_currency(row['amount'])}</div>
                        <div><strong style="color: {COLORS['text_muted']};">Mechanism:</strong> {row['corridor_type']}</div>
                        <div><strong style="color: {COLORS['text_muted']};">Step:</strong> {format_step_time(row['step'])}</div>
                        <div><strong style="color: {COLORS['text_muted']};">Origin:</strong> {row['name_orig_masked']}</div>
                        <div><strong style="color: {COLORS['text_muted']};">Destination:</strong> {row['name_dest_masked']}</div>
                        <div><strong style="color: {COLORS['text_muted']};">Calibrated Prob:</strong> {row['calibrated_probability']:.4f}</div>
                        <div><strong style="color: {COLORS['text_muted']};">Priority Score:</strong> {row['risk_weighted_investigation_priority']:.2f}</div>
                        <div><strong style="color: {COLORS['text_muted']};">Action Code:</strong> {row['action_code']}</div>
                        <div><strong style="color: {COLORS['text_muted']};">Inference Latency:</strong> {row['latency_ms']:.2f} ms</div>
                    </div>
                    <div>
                        <strong style="color: {COLORS['text_muted']}; font-size: 0.85rem;">Active Reason Codes:</strong><br>
                        {render_reason_pills_html(row['decision_reasons'])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with d_col2:
            st.markdown("#### Simulated Analyst Disposition")
            st.caption("⚠️ **Session-only simulated action:** Updates are held in-memory for this analyst session. Does not mutate the persistent database or connect to live core banking clearing.")
            current_status = st.session_state.analyst_case_statuses.get(sel_req_id, "PENDING_REVIEW")
            new_status = st.selectbox(
                "Simulated Status:",
                ["PENDING_REVIEW", "UNDER_INVESTIGATION", "CONFIRMED_FRAUD_ESCALATED", "RESOLVED_LEGITIMATE_RELEASE"],
                index=["PENDING_REVIEW", "UNDER_INVESTIGATION", "CONFIRMED_FRAUD_ESCALATED", "RESOLVED_LEGITIMATE_RELEASE"].index(current_status),
            )
            analyst_notes = st.text_area("Investigation Rationale Notes:", placeholder="Document evidence or suspicious patterns...")
            if st.button("Save Simulated Case Update", use_container_width=True):
                st.session_state.analyst_case_statuses[sel_req_id] = new_status
                st.success(f"Simulated case status updated to: {new_status} (session-only)")


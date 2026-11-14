"""Module C: Single Transaction Deep Explorer & Explainability Card."""

import uuid
import streamlit as st
from dashboard.config.theme import COLORS, get_institutional_css
from dashboard.components.cards import render_kpi_card, render_disclaimer_banner, get_risk_badge_html, render_reason_pills_html
from dashboard.components.charts import plot_shap_horizontal
from dashboard.components.sidebar import render_sidebar
from dashboard.services.api_client import api_client
from dashboard.services.audit_service import audit_service
from dashboard.utils.formatters import format_currency, format_latency, format_probability, format_step_time, mask_account

st.set_page_config(
    page_title="Transaction Explorer | Fraud Risk Intelligence",
    page_icon="🔎",
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
            Transaction Explorer & Fraud Risk Explanation Card
        </h1>
        <p style="font-size: 0.88rem; color: {COLORS['text_muted']}; margin: 4px 0 0 0;">
            Deep-dive transaction scoring, calibrated probability decomposition, and local TreeSHAP attribution.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
render_disclaimer_banner()

tab_score, tab_inspect = st.tabs(["⚡ Real-Time Scoring Console (FastAPI)", "📁 Inspect Persisted Audit Record"])

with tab_score:
    engine_ready = api_client.check_ready().get("status") == "ready"
    if not engine_ready:
        st.warning(
            "⚡ **DEMO MODE: FastAPI Scoring Engine is OFFLINE.** "
            "Interactive transaction scoring is temporarily disabled. Inspect pre-scored benchmark records and TreeSHAP cards in the "
            "**'📁 Inspect Persisted Audit Record'** tab or start the FastAPI service (`uvicorn src.serving.app:app --port 8000`)."
        )

    st.markdown("### Dispatch Transaction to FastAPI Scoring Engine")
    st.caption("Submits transaction payload to POST /v1/score with TreeSHAP explanation enabled.")

    with st.form("score_transaction_form"):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            req_id_input = st.text_input("Request ID:", value=f"workbench-{uuid.uuid4().hex[:8]}")
            step_input = st.number_input("PaySim Step (1–743):", min_value=1, max_value=743, value=120)
            corridor_input = st.selectbox("Transaction Mechanism (Type):", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], index=0)
        with f_col2:
            amount_input = st.number_input("Transaction Amount ($):", min_value=0.01, max_value=100_000_000.0, value=250_000.0, step=1000.0)
            orig_acc_input = st.text_input("Origin Account:", value="C1234987650")
            old_org_input = st.number_input("Origin Old Balance ($):", min_value=0.0, max_value=100_000_000.0, value=250_000.0, step=1000.0)
        with f_col3:
            dest_acc_input = st.text_input("Destination Account:", value="C9876123450")
            old_dest_input = st.number_input("Destination Old Balance ($):", min_value=0.0, max_value=100_000_000.0, value=0.0, step=1000.0)
            cold_start_input = st.checkbox("Allow Cold Start (Default History)", value=True)

        submit_score = st.form_submit_button(
            "Evaluate Fraud Risk & Generate Explanation Card" if engine_ready else "FastAPI Engine Offline (Demo Mode - Live Scoring Disabled)",
            use_container_width=True,
            disabled=not engine_ready,
        )

    if submit_score:
        payload = {
            "request_id": req_id_input.strip(),
            "step": int(step_input),
            "type": corridor_input,
            "amount": float(amount_input),
            "nameOrig": orig_acc_input.strip(),
            "oldbalanceOrg": float(old_org_input),
            "nameDest": dest_acc_input.strip(),
            "oldbalanceDest": float(old_dest_input),
            "allow_cold_start": cold_start_input,
            "include_explanation": True,
        }

        with st.spinner("Scoring via FastAPI inference engine..."):
            score_resp = api_client.score_transaction(payload)

        if not score_resp.get("_success", False):
            st.error(f"Scoring request failed: {score_resp.get('error', 'Unknown API Error')}")
        else:
            st.success(f"Transaction successfully evaluated by FastAPI engine (Inference Latency: {score_resp.get('latency_ms', 0):.2f} ms).")
            
            # Display Decision Card
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            with res_col1:
                render_kpi_card("Risk Decision", score_resp.get("risk_decision", score_resp.get("decision", "UNKNOWN")), subtitle=f"Risk Score: {score_resp.get('risk_score', 0):.1f} / 100", border_color=COLORS["decline"] if score_resp.get("decision") == "DECLINE" else COLORS["approve"])
            with res_col2:
                render_kpi_card("Operational Recommendation", score_resp.get("operational_recommendation", "N/A"), subtitle=f"Risk Tier: {score_resp.get('risk_tier', 'N/A')}", border_color=COLORS["accent_blue"])
            with res_col3:
                render_kpi_card("Calibrated Probability", format_probability(score_resp.get("calibrated_probability")), subtitle=f"Raw Model Probability: {format_probability(score_resp.get('raw_probability'))}", border_color=COLORS["review"])
            with res_col4:
                render_kpi_card("Investigation Priority", f"{score_resp.get('risk_weighted_investigation_priority', 0.0):.2f}", subtitle="P_cal * ln(1 + amount)", border_color=COLORS["accent_cyan"])

            st.markdown("<br>", unsafe_allow_html=True)

            # Details & Reason Codes
            d_col1, d_col2 = st.columns([1, 1])
            with d_col1:
                st.markdown("#### Transaction Evaluation Context")
                st.markdown(
                    f"""
                    <div class="risk-card" style="font-size: 0.85rem; line-height: 1.6;">
                        <div><strong>Request Correlation ID:</strong> <code>{score_resp.get('request_id')}</code></div>
                        <div><strong>Transaction Token:</strong> <code>{score_resp.get('transaction_id')}</code></div>
                        <div><strong>Origin Account (Masked):</strong> {mask_account(orig_acc_input)}</div>
                        <div><strong>Destination Account (Masked):</strong> {mask_account(dest_acc_input)}</div>
                        <div><strong>Nominal Amount:</strong> {format_currency(amount_input)}</div>
                        <div><strong>Temporal Horizon:</strong> {format_step_time(int(step_input))}</div>
                        <div><strong>Model Version:</strong> {score_resp.get('model_version', '1.0.0')}</div>
                        <div><strong>Cold-Start Applied:</strong> {score_resp.get('cold_start_applied', True)}</div>
                        <div><strong>Persistence Engine:</strong> SQLite WAL mode (Asynchronous)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with d_col2:
                st.markdown("#### Active Reason Codes & Evidentiary Rationale")
                reasons = score_resp.get("decision_reasons", [])
                st.markdown(render_reason_pills_html(reasons), unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.info(
                    f"**Simulated Operational Guidance:** {score_resp.get('operational_recommendation')}. "
                    "Safety guardrail: Prohibited from triggering real clearing or automated client friction."
                )

            # TreeSHAP Attribution Card
            expl = score_resp.get("explanation")
            if expl:
                st.markdown("<hr style='border: none; border-top: 1px solid #3A506B; margin: 25px 0 20px 0;'>", unsafe_allow_html=True)
                st.markdown("### Fraud Risk Explanation Card (Level-1 Risk Operations)")
                st.markdown(f"> **Investigative Narrative:** {expl.get('narrative', 'N/A')}")
                st.caption(
                    "Attribution reflects statistical TreeSHAP log-odds correlation, not causal determination. "
                    "Does not constitute an automated Suspicious Activity Report (SAR)."
                )

                top_risk = expl.get("top_risk_factors", [])
                top_mit = expl.get("top_mitigating_factors", [])
                fig_shap = plot_shap_horizontal(top_risk, top_mit)
                st.plotly_chart(fig_shap, use_container_width=True)

with tab_inspect:
    st.markdown("### Inspect Persisted Audit Record by Request ID")
    recent_logs = audit_service.get_recent_logs(limit=500)
    if recent_logs.empty:
        st.warning("No audit records found in database.")
    else:
        req_id_to_lookup = st.selectbox("Select Request ID from Audit Database:", recent_logs["request_id"].tolist())
        if req_id_to_lookup:
            rec = audit_service.get_record_by_request_id(req_id_to_lookup)
            if rec:
                st.markdown(
                    f"""
                    <div class="risk-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span style="font-size: 1.15rem; font-weight: 700; color: {COLORS['text_main']};">
                                Request: <code>{rec['request_id']}</code>
                            </span>
                            <span>{get_risk_badge_html(rec['decision'])}</span>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-size: 0.85rem;">
                            <div><strong>Amount:</strong> {format_currency(rec['amount'])}</div>
                            <div><strong>Mechanism:</strong> {rec['corridor_type']}</div>
                            <div><strong>Step:</strong> {rec['step']}</div>
                            <div><strong>Calibrated Prob:</strong> {rec['calibrated_probability']:.4f}</div>
                            <div><strong>Origin:</strong> {rec['name_orig_masked']}</div>
                            <div><strong>Destination:</strong> {rec['name_dest_masked']}</div>
                            <div><strong>Recommendation:</strong> {rec['action_code']}</div>
                            <div><strong>Priority Score:</strong> {rec['risk_weighted_investigation_priority']:.2f}</div>
                        </div>
                        <div style="margin-top: 12px;">
                            <strong>Reason Codes:</strong><br>
                            {render_reason_pills_html(rec['decision_reasons'])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if rec.get("explanation"):
                    st.markdown("#### Persisted SHAP Explanation Card")
                    st.markdown(f"> {rec['explanation'].get('narrative')}")
                    top_risk = rec["explanation"].get("top_risk_factors", [])
                    top_mit = rec["explanation"].get("top_mitigating_factors", [])
                    fig_shap = plot_shap_horizontal(top_risk, top_mit)
                    st.plotly_chart(fig_shap, use_container_width=True)


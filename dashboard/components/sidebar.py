"""Sidebar navigation, system health probes, and model lineage status."""

import streamlit as st
from dashboard.config.settings import settings
from dashboard.config.theme import COLORS
from dashboard.services.api_client import api_client
from dashboard.services.audit_service import audit_service


def render_sidebar() -> None:
    """Renders persistent sidebar header, status probes, and system indicators."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 10px 0 15px 0; border-bottom: 1px solid {COLORS['border']};">
                <div style="font-size: 0.75rem; font-weight: 700; color: {COLORS['accent_cyan']}; letter-spacing: 0.08em; text-transform: uppercase;">
                    JPMorgan Chase & Co.
                </div>
                <div style="font-size: 1.15rem; font-weight: 700; color: {COLORS['text_main']}; margin-top: 2px;">
                    Fraud Risk Workbench
                </div>
                <div style="font-size: 0.72rem; color: {COLORS['text_dim']}; margin-top: 4px;">
                    {settings.app_version}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.8rem; font-weight: 600; color: {COLORS['text_muted']}; margin-bottom: 8px;'>SYSTEM INTEGRITY PROBES</div>", unsafe_allow_html=True)

        # 1. API Health Check
        ready_res = api_client.check_ready()
        api_status = ready_res.get("status", "offline")
        if api_status == "ready":
            api_badge = f"<span style='color: {COLORS['approve']}; font-weight: 600;'>● ONLINE (Ready)</span>"
        elif api_status == "degraded":
            api_badge = f"<span style='color: {COLORS['review']}; font-weight: 600;'>● DEGRADED</span>"
        else:
            api_badge = f"<span style='color: {COLORS['decline']}; font-weight: 600;'>○ OFFLINE</span>"

        # 2. Database Status
        audit_summary = audit_service.get_audit_summary()
        db_status = audit_summary.get("status", "UNAVAILABLE")
        if "ACTIVE" in db_status:
            db_badge = f"<span style='color: {COLORS['approve']}; font-weight: 600;'>● SQLite (WAL)</span>"
        else:
            db_badge = f"<span style='color: {COLORS['review']}; font-weight: 600;'>○ {db_status}</span>"

        st.markdown(
            f"""
            <div style="background-color: {COLORS['bg_surface']}; border: 1px solid {COLORS['border']}; border-radius: 6px; padding: 10px 12px; font-size: 0.78rem; margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: {COLORS['text_muted']};">FastAPI Scoring:</span>
                    {api_badge}
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color: {COLORS['text_muted']};">Audit Persistence:</span>
                    {db_badge}
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: {COLORS['text_muted']};">Persisted Records:</span>
                    <span style="color: {COLORS['text_main']}; font-weight: 600;">{audit_summary.get('total_count', 0):,}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3. Model Lineage Status
        st.markdown(f"<div style='font-size: 0.8rem; font-weight: 600; color: {COLORS['text_muted']}; margin-bottom: 8px;'>FROZEN MODEL LINEAGE</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background-color: {COLORS['bg_surface']}; border: 1px solid {COLORS['border']}; border-radius: 6px; padding: 10px 12px; font-size: 0.75rem; line-height: 1.4; color: {COLORS['text_muted']};">
                <div><strong>Model:</strong> PaySim_CostWeightedXGB</div>
                <div><strong>Calibration:</strong> Isotonic Regression</div>
                <div><strong>Thresholds:</strong> &tau;<sub>low</sub>=0.10 | &tau;<sub>high</sub>=0.75</div>
                <div><strong>Features:</strong> 31 Pre-decision Inputs</div>
                <div><strong>SHAP Parity:</strong> Verified TreeSHAP</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            "Quantitative Research Prototype on PaySim Benchmark. "
            "Simulated recommendations prohibited from triggering live payment clearing."
        )


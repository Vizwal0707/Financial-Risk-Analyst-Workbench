"""Reusable KPI cards, risk badges, and visual tags for the risk workbench."""

from typing import List, Optional
import streamlit as st
from dashboard.config.theme import COLORS


def render_kpi_card(
    title: str,
    value: str,
    subtitle: Optional[str] = None,
    source: Optional[str] = None,
    border_color: Optional[str] = None,
) -> None:
    """Renders a styled institutional KPI metric card with source annotation."""
    b_color = border_color or COLORS["border"]
    sub_html = f"<div class='risk-card-sub'>{subtitle}</div>" if subtitle else ""
    src_html = f"<div style='font-size: 0.70rem; color: {COLORS['text_dim']}; margin-top: 6px; font-style: italic;'>Source: {source}</div>" if source else ""

    html = f"""
    <div class="risk-card" style="border-left: 4px solid {b_color};">
        <div class="risk-card-header">{title}</div>
        <div class="risk-card-value">{value}</div>
        {sub_html}
        {src_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_risk_badge_html(decision: str) -> str:
    """Returns HTML for inline colored decision badge."""
    dec = str(decision).upper()
    if dec in ("APPROVE", "STP_APPROVE"):
        return f"<span class='badge-approve'>APPROVE</span>"
    elif dec in ("REVIEW", "QUEUE_L1_REVIEW"):
        return f"<span class='badge-review'>REVIEW</span>"
    elif dec in ("DECLINE", "BLOCK_TRANSACTION", "REQUIRE_STEP_UP_AUTHENTICATION"):
        return f"<span class='badge-decline'>DECLINE</span>"
    return f"<span style='color: {COLORS['text_muted']};'>{decision}</span>"


def render_reason_pills_html(reasons: List[str]) -> str:
    """Formats reason codes into compact terminal-style pills."""
    if not reasons:
        return f"<span style='color: {COLORS['text_dim']}; font-size: 0.75rem;'>None</span>"
    pills = [f"<span class='reason-pill'>{r}</span>" for r in reasons]
    return "".join(pills)


def render_disclaimer_banner(
    text: str = (
        "Simulated financial risk intelligence environment evaluated on the PaySim benchmark. "
        "Strictly prohibited from triggering live payment authorization, settlement, or banking rails."
    ),
) -> None:
    """Renders persistent governance safety disclaimer banner."""
    html = f"""
    <div class="disclaimer-banner">
        <strong>LEGAL & GOVERNANCE NOTICE:</strong> {text}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


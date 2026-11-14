"""Components package for dashboard."""

from dashboard.components.cards import (
    get_risk_badge_html,
    render_disclaimer_banner,
    render_kpi_card,
    render_reason_pills_html,
)
from dashboard.components.charts import (
    plot_corridor_distribution,
    plot_decision_donut,
    plot_shap_horizontal,
    plot_step_timeline,
)
from dashboard.components.sidebar import render_sidebar
from dashboard.components.tables import render_ablation_table, render_investigation_table

__all__ = [
    "render_kpi_card",
    "get_risk_badge_html",
    "render_reason_pills_html",
    "render_disclaimer_banner",
    "plot_decision_donut",
    "plot_corridor_distribution",
    "plot_step_timeline",
    "plot_shap_horizontal",
    "render_investigation_table",
    "render_ablation_table",
    "render_sidebar",
]


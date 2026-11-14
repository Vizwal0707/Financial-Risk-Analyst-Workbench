"""Dashboard configuration package."""

from dashboard.config.settings import settings
from dashboard.config.theme import COLORS, PLOTLY_TEMPLATE, get_institutional_css

__all__ = ["settings", "COLORS", "PLOTLY_TEMPLATE", "get_institutional_css"]


"""Institutional visual design, color tokens, and Plotly templates."""

import plotly.graph_objects as go
import plotly.io as pio

# Institutional Color Palette
COLORS = {
    "bg_dark": "#0B132B",
    "bg_card": "#1C2541",
    "bg_surface": "#243254",
    "border": "#3A506B",
    "border_light": "#4A6282",
    "text_main": "#F8FAFC",
    "text_muted": "#94A3B8",
    "text_dim": "#64748B",
    # Risk Indicators
    "approve": "#10B981",  # Emerald
    "approve_bg": "rgba(16, 185, 129, 0.15)",
    "review": "#F59E0B",  # Amber
    "review_bg": "rgba(245, 158, 11, 0.15)",
    "decline": "#EF4444",  # Crimson
    "decline_bg": "rgba(239, 68, 68, 0.15)",
    # Primary & Secondary Accents
    "accent_cyan": "#00B4D8",
    "accent_blue": "#3B82F6",
    "accent_purple": "#8B5CF6",
    "accent_teal": "#14B8A6",
}

# Plotly Theme Definition
PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS["bg_card"],
        plot_bgcolor=COLORS["bg_card"],
        font={"family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif", "color": COLORS["text_main"], "size": 12},
        title={"font": {"size": 15, "color": COLORS["text_main"]}, "x": 0.02, "xanchor": "left"},
        xaxis={
            "gridcolor": COLORS["border"],
            "linecolor": COLORS["border"],
            "zerolinecolor": COLORS["border"],
            "tickfont": {"color": COLORS["text_muted"], "size": 11},
            "title": {"font": {"color": COLORS["text_muted"], "size": 12}},
        },
        yaxis={
            "gridcolor": COLORS["border"],
            "linecolor": COLORS["border"],
            "zerolinecolor": COLORS["border"],
            "tickfont": {"color": COLORS["text_muted"], "size": 11},
            "title": {"font": {"color": COLORS["text_muted"], "size": 12}},
        },
        legend={
            "font": {"color": COLORS["text_muted"], "size": 11},
            "bgcolor": "rgba(28, 37, 65, 0.7)",
            "bordercolor": COLORS["border"],
            "borderwidth": 1,
        },
        margin={"l": 45, "r": 25, "t": 45, "b": 45},
    )
)
pio.templates["institutional_dark"] = PLOTLY_TEMPLATE
pio.templates.default = "institutional_dark"


def get_institutional_css() -> str:
    """Returns CSS to apply institutional financial workbench styling to Streamlit."""
    return f"""
    <style>
    /* Global Base */
    .stApp {{
        background-color: {COLORS['bg_dark']};
        color: {COLORS['text_main']};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    
    /* Header & Navigation Bar */
    header[data-testid="stHeader"] {{
        background-color: rgba(11, 19, 43, 0.85);
        backdrop-filter: blur(8px);
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['bg_card']};
        border-right: 1px solid {COLORS['border']};
    }}
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
        color: {COLORS['accent_cyan']};
    }}
    
    /* Card Container */
    .risk-card {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
    }}
    .risk-card-header {{
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {COLORS['text_muted']};
        margin-bottom: 6px;
    }}
    .risk-card-value {{
        font-size: 1.85rem;
        font-weight: 700;
        color: {COLORS['text_main']};
        line-height: 1.2;
    }}
    .risk-card-sub {{
        font-size: 0.78rem;
        color: {COLORS['text_dim']};
        margin-top: 4px;
    }}

    /* Badges */
    .badge-approve {{
        background-color: {COLORS['approve_bg']};
        color: {COLORS['approve']};
        border: 1px solid {COLORS['approve']};
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }}
    .badge-review {{
        background-color: {COLORS['review_bg']};
        color: {COLORS['review']};
        border: 1px solid {COLORS['review']};
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }}
    .badge-decline {{
        background-color: {COLORS['decline_bg']};
        color: {COLORS['decline']};
        border: 1px solid {COLORS['decline']};
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.75rem;
    }}
    
    /* Reason Code Pill */
    .reason-pill {{
        display: inline-block;
        background-color: {COLORS['bg_surface']};
        color: {COLORS['accent_cyan']};
        border: 1px solid {COLORS['border_light']};
        padding: 2px 7px;
        border-radius: 4px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.74rem;
        margin: 2px 4px 2px 0;
    }}

    /* Institutional Disclaimer Notice */
    .disclaimer-banner {{
        background-color: rgba(36, 50, 84, 0.4);
        border-left: 3px solid {COLORS['accent_cyan']};
        padding: 10px 14px;
        border-radius: 0 4px 4px 0;
        font-size: 0.78rem;
        color: {COLORS['text_muted']};
        margin-bottom: 18px;
    }}

    /* Streamlit Tables & DataFrames */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
    }}
    </style>
    """

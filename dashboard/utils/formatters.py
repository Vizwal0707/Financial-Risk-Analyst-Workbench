"""Formatting and sanitization utilities for the financial risk workbench."""

from typing import Any, Optional


def format_currency(val: Optional[float], decimals: int = 2) -> str:
    """Formats monetary amounts into institutional currency representation ($X,XXX.XX)."""
    if val is None:
        return "$0.00"
    try:
        f_val = float(val)
        if abs(f_val) >= 1_000_000_000:
            return f"${f_val / 1_000_000_000:,.2f}B"
        if abs(f_val) >= 1_000_000:
            return f"${f_val / 1_000_000:,.2f}M"
        return f"${f_val:,.{decimals}f}"
    except (ValueError, TypeError):
        return "$0.00"


def format_percentage(val: Optional[float], decimals: int = 2) -> str:
    """Formats numeric ratio [0.0, 1.0] as a percentage string (XX.XX%)."""
    if val is None:
        return "0.00%"
    try:
        return f"{float(val) * 100.0:.{decimals}f}%"
    except (ValueError, TypeError):
        return "0.00%"


def format_probability(val: Optional[float], decimals: int = 4) -> str:
    """Formats model probability [0.0, 1.0] with scientific precision."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"


def format_step_time(step: Optional[int]) -> str:
    """Translates PaySim simulation step (1 step = 1 hour) into Day & Hour format."""
    if step is None or step < 1:
        return "Step 0 (Day 1, 00:00)"
    day = int((step - 1) // 24) + 1
    hour = int((step - 1) % 24)
    return f"Step {step} (Day {day}, {hour:02d}:00)"


def mask_account(account_id: Any) -> str:
    """Masks customer/account numbers for PCI-DSS-aligned account masking best practice (non-certified prototype)."""
    if account_id is None:
        return "****"
    acc = str(account_id).strip()
    if len(acc) <= 4:
        return "****"
    return f"{acc[:2]}****{acc[-4:]}"


def format_latency(ms: Optional[float]) -> str:
    """Formats inference latency in milliseconds or microseconds."""
    if ms is None:
        return "0.0 ms"
    try:
        val = float(ms)
        if val < 1.0:
            return f"{val * 1000.0:.0f} μs"
        return f"{val:.2f} ms"
    except (ValueError, TypeError):
        return "0.0 ms"


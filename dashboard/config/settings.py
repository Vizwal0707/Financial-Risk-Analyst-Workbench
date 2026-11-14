"""Configuration settings for the Risk Analyst Workbench & Governance Dashboard."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DashboardSettings:
    """Central configuration for dashboard API and persistence endpoints."""

    # API Configuration
    api_base_url: str = os.getenv("FRAUD_API_URL", "http://127.0.0.1:8000")
    api_timeout_seconds: float = float(os.getenv("FRAUD_API_TIMEOUT", "5.0"))

    # Database Configuration
    database_path: str = os.getenv("FRAUD_DB_PATH", "data/audit_scoring.db")

    # Evaluation Artifacts
    model_comparison_path: str = "reports/model_comparison.json"
    test_metrics_path: str = "reports/test_evaluation_metrics.json"
    confusion_matrix_path: str = "reports/decision_confusion_matrix.csv"
    feature_ablation_path: str = "reports/feature_ablation_results.csv"

    # Default Policy Baseline
    default_tau_low: float = 0.10
    default_tau_high: float = 0.75
    default_c_fp: float = 100.0
    default_c_rev: float = 25.0
    default_daily_capacity: int = 500

    # Application Metadata
    app_title: str = "JPMorgan Chase & Co. | Financial Crime Risk Intelligence"
    app_subtitle: str = "Institutional Fraud Risk Workbench & Model Governance Dashboard"
    app_version: str = "1.0.0 (Phase 6)"
    evaluation_benchmark: str = "PaySim Synthetic Benchmark (6,362,620 transactions)"


settings = DashboardSettings()


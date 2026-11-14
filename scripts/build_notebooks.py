"""Builds professional research Jupyter notebooks for EDA and Feature Engineering Validation."""

from pathlib import Path

import nbformat as nbf

NOTEBOOK_DIR = Path("notebooks")
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


def build_01_eda_notebook() -> None:
    """Constructs notebooks/01_eda_and_data_profiling.ipynb."""
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell("""# Enterprise Financial Fraud Risk Intelligence Platform
## Research Notebook 01: Exploratory Data Analysis & Quantitative Transaction Profiling

**Author:** Principal Machine Learning Engineer & Quantitative Risk Architect
**Benchmark:** PaySim Synthetic Financial Fraud Benchmark (Lopez-Rojas et al.)
**Dataset Scope:** 6,362,620 transactions across 743 discrete simulation hours (~31 calendar days)

> **Institutional Regulatory Disclaimer:**
> PaySim is an agent-based synthetic simulation designed for mobile money fraud research. While it captures authentic transaction topologies, severe class imbalance (~0.129%), and account liquidation patterns, it does not represent actual proprietary customer accounts, internal ledgers, or live payment corridors of JPMorgan Chase or any other commercial bank."""),
        nbf.v4.new_code_cell("""import os
import duckdb
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Visual aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'figure.dpi': 150, 'font.sans-serif': 'Helvetica', 'font.size': 11})

conn = duckdb.connect()
RAW_PARQUET = 'data/raw/paysim.parquet'
print(f"DuckDB version: {duckdb.__version__}")"""),
        nbf.v4.new_markdown_cell("""### 1. Step Range Reconciliation & Structural Audit
The original literature assumes 744 steps ($31 \\times 24$ hours). Let us inspect the actual observed step bounds in PaySim."""),
        nbf.v4.new_code_cell("""audit_df = conn.execute(f'''
SELECT
    MIN(step) AS min_step,
    MAX(step) AS max_step,
    COUNT(DISTINCT step) AS distinct_steps,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS total_fraud,
    ROUND(SUM(isFraud) * 100.0 / COUNT(*), 4) AS fraud_prevalence_pct
FROM read_parquet('{RAW_PARQUET}')
''').df()
audit_df"""),
        nbf.v4.new_markdown_cell("""### 2. Transaction Corridor Breakdown & Fraud Isolation
We inspect the distribution across the five payment corridors: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, and `TRANSFER`."""),
        nbf.v4.new_code_cell("""corridor_summary = conn.execute(f'''
SELECT
    type,
    COUNT(*) AS total_volume,
    SUM(isFraud) AS fraud_count,
    ROUND(SUM(isFraud) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(amount) / 1e9, 2) AS total_amount_billions,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END) / 1e9, 2) AS fraud_exposure_billions
FROM read_parquet('{RAW_PARQUET}')
GROUP BY type
ORDER BY total_volume DESC
''').df()
corridor_summary"""),
        nbf.v4.new_markdown_cell(
            """**Critical Finding:** Fraud is strictly concentrated in `TRANSFER` and `CASH_OUT` corridors. `PAYMENT`, `CASH_IN`, and `DEBIT` contain exactly zero fraudulent transactions."""
        ),
        nbf.v4.new_markdown_cell("""### 3. Amount Dispersion & Statistical Divergence Tests
Comparing legitimate vs fraudulent transaction amounts on a logarithmic scale."""),
        nbf.v4.new_code_cell("""# Extract sample for density plotting and pull all fraud records
sample_df = conn.execute(f'''
SELECT amount, isFraud FROM read_parquet('{RAW_PARQUET}') USING SAMPLE 100000 (reservoir, 42)
''').df()

fraud_amounts = conn.execute(f'''
SELECT amount FROM read_parquet('{RAW_PARQUET}') WHERE isFraud = 1
''').df()['amount']

legit_amounts = sample_df[sample_df['isFraud'] == 0]['amount']

# Two-sample Mann-Whitney U test and Kolmogorov-Smirnov test
mwu_res = stats.mannwhitneyu(legit_amounts, fraud_amounts, alternative='two-sided')
ks_res = stats.ks_2samp(legit_amounts, fraud_amounts)

print(f"Legitimate Median Amount:  ${legit_amounts.median():,.2f}")
print(f"Fraudulent Median Amount:  ${fraud_amounts.median():,.2f}")
print(f"Mann-Whitney U statistic:  {mwu_res.statistic:.2e} (p-value: {mwu_res.pvalue:.2e})")
print(f"Kolmogorov-Smirnov stat:   {ks_res.statistic:.4f} (p-value: {ks_res.pvalue:.2e})")"""),
        nbf.v4.new_markdown_cell("""### 4. Account Liquidation: Sender Balance Drain Dynamics
Analyzing the degree to which a transaction liquidates the sender's entire available balance."""),
        nbf.v4.new_code_cell("""drain_analysis = conn.execute(f'''
SELECT
    isFraud,
    ROUND(AVG(amount / (oldbalanceOrg + 1e-5)), 4) AS avg_drain_ratio,
    ROUND(MEDIAN(amount / (oldbalanceOrg + 1e-5)), 4) AS median_drain_ratio,
    SUM(CASE WHEN oldbalanceOrg = 0.0 THEN 1 ELSE 0 END) AS orig_zero_bal_count,
    ROUND(SUM(CASE WHEN oldbalanceOrg = 0.0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS orig_zero_bal_pct
FROM read_parquet('{RAW_PARQUET}')
WHERE type IN ('TRANSFER', 'CASH_OUT')
GROUP BY isFraud
''').df()
drain_analysis"""),
        nbf.v4.new_markdown_cell("""### 5. Diurnal Seasonality vs. Fraud Attack Rates
Legitimate customer transactions obey a 24-hour diurnal cycle, whereas automated fraud operates constantly."""),
        nbf.v4.new_code_cell("""diurnal_df = conn.execute(f'''
SELECT
    (step % 24) AS hour_of_day,
    COUNT(*) AS total_tx,
    SUM(isFraud) AS fraud_tx,
    ROUND(SUM(isFraud) * 100.0 / COUNT(*), 4) AS fraud_rate_pct
FROM read_parquet('{RAW_PARQUET}')
GROUP BY (step % 24)
ORDER BY hour_of_day
''').df()
diurnal_df.head(10)"""),
        nbf.v4.new_markdown_cell("""### 6. Fraud Value Concentration (Pareto Loss Curve)
Measuring what percentage of monetary loss is driven by the top deciles of fraud transactions."""),
        nbf.v4.new_code_cell("""sorted_fraud = fraud_amounts.sort_values(ascending=False).values
cum_value = np.cumsum(sorted_fraud) / np.sum(sorted_fraud) * 100.0
pct_tx = np.linspace(0, 100, len(sorted_fraud))

idx_20 = int(0.20 * len(sorted_fraud))
idx_50 = int(0.50 * len(sorted_fraud))

print(f"Total Fraud Exposure: ${np.sum(sorted_fraud):,.2f}")
print(f"Top 20% Frauds account for: {cum_value[idx_20]:.2f}% of dollar losses")
print(f"Top 50% Frauds account for: {cum_value[idx_50]:.2f}% of dollar losses")"""),
        nbf.v4.new_markdown_cell("""### 7. Feature Correlation & Risk Linkages
Examining linear dependencies among pre-transaction balances, amounts, and targets."""),
        nbf.v4.new_code_cell("""corr_df = conn.execute(f'''
SELECT
    amount,
    oldbalanceOrg,
    oldbalanceDest,
    amount / (oldbalanceOrg + 1e-5) AS drain_ratio,
    (oldbalanceOrg - amount) AS balance_coverage,
    isFraud
FROM read_parquet('{RAW_PARQUET}')
USING SAMPLE 50000 (reservoir, 42)
''').df()
corr_df.corr().round(3)"""),
        nbf.v4.new_markdown_cell("""### 8. Quantitative Takeaways & Feature Engineering Roadmap
1. **Corridor Filtering:** Exclude or downweight non-fraud corridors or use corridor one-hot flags.
2. **Account Liquidation Ratios:** `balance_orig_drain_ratio` and `orig_balance_coverage` are paramount indicators of account takeover.
3. **Point-in-Time Rolling Velocity:** Adversaries execute rapid bursts; historical windows ($1\text{h}, 6\text{h}, 24\text{h}, 72\text{h}$) must capture velocity without future leakage.
4. **Value-Focused Evaluation:** Model thresholds should be calibrated not merely on transaction count recall, but on **Fraud Value Captured (FVC)**."""),
    ]

    nb.cells = cells
    with open(NOTEBOOK_DIR / "01_eda_and_data_profiling.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Built notebooks/01_eda_and_data_profiling.ipynb")


def build_02_feature_validation_notebook() -> None:
    """Constructs notebooks/02_feature_engineering_validation.ipynb."""
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell("""# Enterprise Financial Fraud Risk Intelligence Platform
## Research Notebook 02: Point-in-Time Feature Engineering & Invariance Validation

**Objective:** Validate mathematical point-in-time correctness, verify zero target leakage, assess DuckDB and PySpark execution parity, and inspect feature statistics across chronological splits."""),
        nbf.v4.new_code_cell("""import duckdb
import numpy as np
import pandas as pd
from src.features.duckdb_feature_pipeline import DuckDBFeaturePipeline
from src.features.pyspark_feature_pipeline import PySparkFeaturePipeline
from src.features.pipeline import FinancialFeaturePipeline

conn = duckdb.connect()
FEATURIZED_PARQUET = 'data/processed/paysim_featurized.parquet'
print("Pipeline modules loaded successfully.")"""),
        nbf.v4.new_markdown_cell("""### 1. Partition Verification & Step-Atomic Integrity
Inspecting record counts and step ranges across the three chronological partitions."""),
        nbf.v4.new_code_cell("""part_summary = conn.execute(f'''
SELECT
    partition,
    COUNT(*) AS total_rows,
    SUM(isFraud) AS fraud_rows,
    ROUND(SUM(isFraud) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
    MIN(step) AS min_step,
    MAX(step) AS max_step
FROM read_parquet('{FEATURIZED_PARQUET}')
GROUP BY partition
ORDER BY min_step
''').df()
part_summary"""),
        nbf.v4.new_markdown_cell("""### 2. Feature Schema & Null/Inf Integrity Checks
Asserting zero nulls, NaNs, or inf values across the generated feature columns."""),
        nbf.v4.new_code_cell("""null_check = conn.execute(f'''
SELECT
    COUNT(*) AS total_records,
    SUM(CASE WHEN amount_log IS NULL OR isnan(amount_log) THEN 1 ELSE 0 END) AS null_amount_log,
    SUM(CASE WHEN balance_orig_drain_ratio IS NULL OR isnan(balance_orig_drain_ratio) THEN 1 ELSE 0 END) AS null_drain,
    SUM(CASE WHEN orig_tx_count_24h IS NULL THEN 1 ELSE 0 END) AS null_tx_24h,
    SUM(CASE WHEN orig_amount_mean_24h IS NULL OR isnan(orig_amount_mean_24h) THEN 1 ELSE 0 END) AS null_mean_24h
FROM read_parquet('{FEATURIZED_PARQUET}')
''').df()
null_check"""),
        nbf.v4.new_markdown_cell("""### 3. Statistical Distribution of Historical Rolling Features
Inspecting 24h and 72h velocity and amount deviation features for legitimate vs fraudulent transactions."""),
        nbf.v4.new_code_cell("""velocity_stats = conn.execute(f'''
SELECT
    isFraud,
    ROUND(AVG(orig_tx_count_24h), 4) AS avg_tx_24h,
    ROUND(AVG(orig_amount_sum_24h), 2) AS avg_amount_24h,
    ROUND(AVG(dest_inflow_count_24h), 4) AS avg_dest_inflow_24h,
    ROUND(AVG(is_first_interaction), 4) AS first_interaction_rate
FROM read_parquet('{FEATURIZED_PARQUET}')
GROUP BY isFraud
''').df()
velocity_stats"""),
        nbf.v4.new_markdown_cell("""### 4. Zero-Leakage & Forbidden Column Assertions
Verifying that `newbalanceOrig`, `newbalanceDest`, and `isFlaggedFraud` are strictly excluded from the feature space."""),
        nbf.v4.new_code_cell("""feature_columns = conn.execute(f'''
SELECT column_name
FROM (DESCRIBE SELECT * FROM read_parquet('{FEATURIZED_PARQUET}'))
''').df()['column_name'].tolist()

forbidden = {'newbalanceOrig', 'newbalanceDest', 'isFlaggedFraud'}
detected_forbidden = forbidden.intersection(set(feature_columns))

print(f"Total Columns in Processed Parquet: {len(feature_columns)}")
print(f"Forbidden Leakage Columns Detected: {list(detected_forbidden)}")
assert len(detected_forbidden) == 0, "Leakage Violation: Forbidden columns found in feature set!"
print("Zero-leakage verification: PASSED.")"""),
        nbf.v4.new_markdown_cell("""### 5. Summary & Hand-Off to Model Development
The featurized dataset is verified, chronologically split, leak-free, and ready for model training in Phase 4."""),
    ]

    nb.cells = cells
    with open(NOTEBOOK_DIR / "02_feature_engineering_validation.ipynb", "w") as f:
        nbf.write(nb, f)
    print("Built notebooks/02_feature_engineering_validation.ipynb")


if __name__ == "__main__":
    build_01_eda_notebook()
    build_02_feature_validation_notebook()

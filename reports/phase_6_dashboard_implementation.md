# Phase 6 Dashboard Implementation & System Architecture Report

**Implementation Date:** 2026-11-15  
**Document Classification:** Engineering Implementation Report (Phase 6 Deliverable)  
**Target System:** Institutional Risk Analyst Workbench & Model Governance Dashboard  
**Technology Stack:** Python 3.11, Streamlit 1.67, Plotly 7.1, Pandas 2.2, FastAPI, SQLite (WAL)  
**Status:** **IMPLEMENTED, VERIFIED, AND FULLY INTEGRATED**

---

## 1. Executive Summary

Phase 6 implements an institutional-grade, interactive web-based Financial Fraud Risk Workbench and Model Governance Dashboard. Built to the operational and visual standards of Tier-1 financial institutions (JPMorgan Chase caliber), this platform provides real-time decisioning visibility, prioritized investigation workflows, local TreeSHAP explainability, regulatory model governance diagnostics, and interactive policy simulation.

### Core Architectural Boundaries
1. **Zero Model Duplication:** The Streamlit dashboard operates purely as an analytical presentation and governance client. It does **not** duplicate, instantiate, or retrain the XGBoost model, probability calibrators, deep autoencoder, or feature builders. All live scoring requests and SHAP attribution cards are consumed via HTTP from the existing FastAPI serving architecture (`POST /v1/score`).
2. **Strict Data Lineage Separation:** A rigorous barrier isolates:
   - **Ground-Truth Historical Test Benchmark ($N=90,829$, $1,366$ frauds):** Static, frozen metrics loaded from Phase 4/4.1 reports (`reports/model_comparison.json`, `reports/test_evaluation_metrics.json`, `reports/decision_confusion_matrix.csv`).
   - **Live Scored Audit Database ($11,571+$ records):** Persisted in SQLite with Write-Ahead Logging (`data/audit_scoring.db`).
   - **Hypothetical Policy Simulations:** Non-destructive in-memory threshold calculations evaluated over offline evaluation datasets without modifying production configurations.
3. **Institutional Visual Design:** Implements a dark navy/charcoal command center palette (`#0B132B`, `#1C2541`, `#3A506B`) with restrained status indicators (emerald for approvals, amber for reviews, crimson for declines), high tabular information density, and custom Plotly styling.
4. **Mandatory Privacy & Governance Guardrails:** Enforces PCI-DSS account masking (`C1****7890`), non-causal explainability framing ("Fraud Risk Explanation Cards"), and mandatory safety disclaimers prohibiting simulated recommendations from triggering live payment rails.

---

## 2. Directory Layout & Modular Structure

```
dashboard/
├── app.py                      # Main landing page & operational overview
├── config/
│   ├── __init__.py
│   ├── settings.py             # Central configuration (URLs, timeouts, database paths)
│   └── theme.py                # Institutional palette, Plotly template, custom CSS
├── utils/
│   ├── __init__.py
│   └── formatters.py           # Currency ($X,XXX.XX), percentages, timestamps, masking
├── services/
│   ├── __init__.py
│   ├── api_client.py           # FastAPI client with timeout & error handling
│   ├── audit_service.py        # Read-only SQLite audit repository reader
│   ├── evaluation_service.py   # Loader for Phase 4/4.1 frozen model evaluation reports
│   └── simulation_service.py   # In-memory policy threshold & financial cost simulator
├── components/
│   ├── __init__.py
│   ├── cards.py                # KPI cards, decision badges, reason-code pills
│   ├── charts.py               # Plotly figures (donut, bar, timeline, heatmap, SHAP)
│   ├── tables.py               # Formatted investigation queue & ablation tables
│   └── sidebar.py              # System health probes, model lineage, navigation
└── pages/
    ├── 01_Executive_Overview.py    # Module A: Portfolio KPIs, fraud rates, trends
    ├── 02_Investigation_Queue.py  # Module B: Priority-ranked triage table
    ├── 03_Transaction_Explorer.py  # Module C: Real-time scoring & TreeSHAP card
    ├── 04_Model_Governance.py      # Module D: PR-AUC, Brier, calibration, ablations
    ├── 05_Policy_Simulator.py      # Module E: Hypothetical threshold optimizer
    └── 06_Audit_Explorer.py        # Module F: Compliance audit log inspector
```

---

## 3. Detailed Module Implementation

### 3.1 Module A — Executive Risk Overview (`01_Executive_Overview.py`)
- **Scope Toggle:** Analysts can switch between the **Historical Offline Test Benchmark ($N=90,829$)** and the **Live Scored Audit Database ($11,571+$ records)**.
- **Source Labeling:** Every KPI card explicitly cites its source (e.g. *"Source: Frozen Out-of-Time Test Set (Steps 624–743)"* vs. *"Source: SQLite audit_scoring_logs (PRAGMA WAL)"*).
- **Executive Visualizations:**
  1. *Operational Decision Distribution:* Plotly donut chart rendering the volumetric split across Straight-Through Approvals ($98.47\%$), Analyst Queue Reviews ($0.06\%$), and Automated Declines ($1.47\%$).
  2. *Corridor Mechanism Breakdown:* Dual-axis chart correlating transaction volume against mean calibrated fraud probability across mechanisms (`TRANSFER`, `CASH_OUT`, `PAYMENT`, `CASH_IN`, `DEBIT`).
  3. *3x3 Decision Reconciliation:* Reconciles exact transaction counts and dollar values across operational triage states against ground-truth labels ($15$ false positives, $1$ missed fraud of $\$399,045.08$).

### 3.2 Module B — Investigation Queue (`02_Investigation_Queue.py`)
- **Risk-Weighted Triage Ordering:** The queue orders transactions by:
  $$\text{Priority Score} = P_{\text{cal}} \times \ln(1 + \text{amount})$$
  Ensures high-probability frauds are investigated immediately without distortion by low-probability massive corporate wires.
- **Multi-Criteria Filtering & Search:** Real-time filtering by Decision (`ALL`, `REVIEW`, `DECLINE`, `APPROVE`), Risk Tier (`HIGH`, `MEDIUM`, `LOW`), Mechanism, and text search across Request ID or masked account numbers.
- **Simulated Analyst Disposition Console:** Analysts can update case status (`PENDING_REVIEW`, `UNDER_INVESTIGATION`, `CONFIRMED_FRAUD_ESCALATED`, `RESOLVED_LEGITIMATE_RELEASE`) and record investigative notes. State is managed in Streamlit session state and clearly marked as simulated, preventing false claims of real bank filings.

### 3.3 Module C — Transaction Explorer (`03_Transaction_Explorer.py`)
- **Live Scoring Integration:** Provides an interactive transaction input form connected to `POST /v1/score` with `include_explanation=True`.
- **Comprehensive Decision Card:** Renders Risk Decision (`APPROVE`, `REVIEW`, `DECLINE`), Operational Recommendation (`STP_APPROVE`, `QUEUE_L1_REVIEW`, `BLOCK_TRANSACTION`, `REQUIRE_STEP_UP_AUTHENTICATION`), Normalized Risk Score ($[0, 100]$), and Priority Score.
- **Local TreeSHAP Attribution Card:** Visualizes top `RISK_INDUCING` (crimson) and `RISK_MITIGATING` (emerald) feature contributions in log-odds scale, accompanied by a non-causal disclaimer.
- **Audit Lookup:** Analysts can also look up any persisted transaction from the SQLite database to inspect historical predictions and cached SHAP cards.

### 3.4 Module D — Model Governance (`04_Model_Governance.py`)
- **Baseline Model Comparison:** Comprehensive table comparing Cost-Weighted XGBoost against Majority Class, Transaction Type Only, Rule Heuristic, Logistic Regression, Random Forest, and Deep Autoencoder.
- **Probability Calibration Diagnostics:** Visualizes the Reliability Diagram comparing uncalibrated XGBoost (Brier score $0.00029$) against Isotonically Calibrated XGBoost (Brier score $0.00027$, ECE $0.00015$).
- **Performance Curves:** Out-of-time Precision-Recall Curve ($\text{PR-AUC} = 0.9979$) and ROC Curve ($\text{ROC-AUC} = 0.9996$).
- **Independent Feature Ablation:** Renders all 11 audited experimental configurations from Phase 4.1.
- **Drift Monitoring State:** Transparently documents that streaming data drift monitoring is unavailable due to the frozen 743-step simulation horizon, explicitly rejecting fabricated drift statistics.

### 3.5 Module E — Policy Simulator (`05_Policy_Simulator.py`)
- **Interactive Threshold & Cost Sliders:**
  - $\tau_{\text{low}} \in [0.01, 0.40]$ (default: $0.10$).
  - $\tau_{\text{high}} \in [0.45, 0.99]$ (default: $0.75$).
  - False Positive Cost $C_{\text{FP}} \in [\$1, \$1000]$ (default: $\$100$).
  - Investigation Review Cost $C_{\text{REV}} \in [\$1, \$500]$ (default: $\$25$).
  - Daily Review Capacity $K_{\text{daily}} \in [50, 2000]$ cases/day (default: $500$).
  - Fraud Loss Multiplier $M_{\text{FN}} \in [1.0, 3.0]$.
- **Vectorized In-Memory Simulation:** Re-evaluates decision boundaries over $90,829$ pre-scored test transactions in $< 15\text{ ms}$.
- **Comparative Impact Table:** Displays side-by-side delta between the Audited Baseline Policy and the Simulated Policy across Approvals, Reviews, Declines, False Positives, Missed Frauds, and Net Financial Savings.
- **Safety Boundary:** Prominently displays disclaimers confirming that simulation is strictly offline and never alters production API configurations.

### 3.6 Module F — Audit Explorer (`06_Audit_Explorer.py`)
- **Compliance Audit Trail:** Displays immutable scoring logs from `data/audit_scoring.db`.
- **Durability Status Bar:** Identifies the active SQLite 3.45 (WAL Mode) engine and indicates that PostgreSQL container integration is in standby.
- **Deep Trace Modal:** Allows full inspection of Request Correlation IDs, creation timestamps, raw/calibrated probabilities, reason codes, and stored TreeSHAP explanation JSON.
- **Privacy Compliance:** Enforces 100% PII masking across origin and destination accounts.

---

## 4. Quality Assurance & Regression Verification

The platform was subjected to complete regression testing:
```bash
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. ./.venv/bin/pytest tests/ -v
```
- **Total Tests Collected:** 92
- **Total Tests Passed:** 92 (100.0% pass rate)
- **Execution Time:** 21.88 seconds
- **Preservation:** 100% of the 76 Phase 1–5.1 tests were preserved unmodified, supplemented by 16 dedicated dashboard unit tests.

---

## 5. Local Startup Instructions

To launch the Financial Fraud Detection & Risk Intelligence Platform locally:

### Step 1: Start the FastAPI Scoring Engine
In Terminal 1 (from repository root):
```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
*Verification:* Navigate to `http://127.0.0.1:8000/docs` to inspect interactive Swagger API documentation.

### Step 2: Start the Streamlit Risk Workbench Dashboard
In Terminal 2 (from repository root):
```bash
source .venv/bin/activate
MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. streamlit run dashboard/app.py --server.port 8501 --server.address 127.0.0.1
```
*Verification:* Open `http://127.0.0.1:8501` in your web browser. The sidebar health probe will display:
- **FastAPI Scoring:** `● ONLINE (Ready)`
- **Audit Persistence:** `● SQLite (WAL)`


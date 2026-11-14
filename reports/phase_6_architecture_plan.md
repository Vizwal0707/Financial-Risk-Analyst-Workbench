# Phase 6 Architecture Plan: Institutional Risk Analyst Workbench & Interactive Governance Dashboard

**Planning Date:** 2026-11-15  
**Document Classification:** Architecture & Technical Design Specification (Phase 6 Deliverable)  
**System Target:** Tier-1 Financial Crime Risk Intelligence Platform (JPMorgan Chase Caliber)  
**Author:** Quantitative Researcher & Financial Risk Architect  
**Status:** **APPROVED FOR IMPLEMENTATION**

---

## 1. Executive Summary & Design Principles

Phase 6 constructs an institutional-grade, interactive web-based Financial Fraud Risk Workbench and Model Governance Dashboard using **Streamlit**, **Plotly**, and **Pandas**, interfacing directly with the existing, audited **FastAPI** serving architecture and persistent audit repository.

### Core Architectural Principles
1. **Zero Model Duplication:** The dashboard is strictly an analytical client and governance console. It does **not** instantiate or retrain the XGBoost model, probability calibrators, deep autoencoder, or feature builders in Streamlit memory. All real-time transaction evaluations, probability calibrations, and SHAP explanations are consumed via HTTP from the existing FastAPI serving engine (`src/api/routes/`).
2. **Explicit Data Lineage Separation:** A rigorous barrier separates:
   - **Historical Benchmark Evaluations:** Static, frozen, out-of-time test metrics ($N=90,829$, $1,366$ frauds) and validation metrics ($N=209,984$, $1,286$ frauds) loaded from audited Phase 4.1 reports (`reports/model_comparison.json`, `reports/test_evaluation_metrics.json`).
   - **Live Production Scoring Audit Logs:** Dynamically retrieved from SQLite Write-Ahead Logging (`data/audit_scoring.db`).
   - **Hypothetical Policy Simulations:** In-memory, non-destructive threshold sensitivity analyses executed on frozen evaluation datasets without altering production thresholds.
3. **High-Density Institutional Visual Aesthetics:** Designed to resemble modern financial risk command centers (e.g. Palantir Foundry, internal Chase risk platforms) with dark navy/charcoal styling (`#0B132B`, `#1C2541`, `#3A506B`), restrained risk indicators (teal for approvals, amber for reviews, crimson for declines), clean tabular density, and responsive layouts.
4. **Strict Security & Privacy Compliance:** Full PII account masking (`C1****7890`), non-causal explainability framing ("Fraud Risk Explanation Cards"), and mandatory safety disclaimers preventing simulated actions from connecting to live payment authorization networks.

---

## 2. System Architecture & Modular Directory Layout

```
financial-fraud-risk-platform/
├── dashboard/
│   ├── app.py                      # Main Streamlit entrance, theme injection, navigation
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py             # API base URL, timeout, database paths, environment
│   │   └── theme.py                # Color palette, Plotly template, CSS styling injection
│   ├── services/
│   │   ├── __init__.py
│   │   ├── api_client.py           # Robust HTTP client for FastAPI endpoints with fallback
│   │   ├── audit_service.py        # Read-only accessor for SQLite audit_scoring.db
│   │   ├── evaluation_service.py   # Loader for Phase 4/4.1 frozen model evaluation artifacts
│   │   └── simulation_service.py   # Policy threshold & cost optimizer for offline simulation
│   ├── components/
│   │   ├── __init__.py
│   │   ├── cards.py                # Metric cards, risk badges, priority pills
│   │   ├── charts.py               # Plotly figures: PR curve, ROC, calibration, distributions
│   │   ├── tables.py               # Formatted investigation and audit tables
│   │   └── sidebar.py              # Health probe, model info display, global navigation
│   ├── pages/
│   │   ├── 01_Executive_Overview.py    # Module A: Portfolio KPIs, fraud rates, temporal trends
│   │   ├── 02_Investigation_Queue.py  # Module B: Priority-ranked transaction triage queue
│   │   ├── 03_Transaction_Explorer.py  # Module C: Single-transaction SHAP card & context
│   │   ├── 04_Model_Governance.py      # Module D: PR-AUC, Brier, calibration, feature ablation
│   │   ├── 05_Policy_Simulator.py      # Module E: Hypothetical threshold & cost sensitivity
│   │   └── 06_Audit_Explorer.py        # Module F: Live persistent audit log inspection
│   └── utils/
│       ├── __init__.py
│       └── formatters.py           # Currency, percentage, timestamp, account masking
```

---

## 3. Integration & Data Contracts

### 3.1 FastAPI Serving Endpoints Consumed
The dashboard communicates with the backend via `dashboard/services/api_client.py`:
- `GET /health`: Liveness probe verifying API uptime.
- `GET /ready`: Readiness probe verifying XGBoost booster, isotonic calibrator, and SQLite reachability.
- `GET /v1/model/info`: Model metadata, 31 active feature names, isotonic calibration method, decision thresholds ($\tau_{\text{low}}=0.10, \tau_{\text{high}}=0.75$), and baseline cost assumptions ($C_{FP}=\$100, C_{REV}=\$25$).
- `POST /v1/score`: Real-time single transaction scoring with optional TreeSHAP card (`include_explanation=True`).
- `POST /v1/score/batch`: Vectorized multi-transaction batch scoring for high-throughput testing.

### 3.2 Read-Only Audit Repository Access
The dashboard queries `data/audit_scoring.db` via `dashboard/services/audit_service.py`:
- SQLite WAL mode is opened with read-only connection URI: `file:data/audit_scoring.db?mode=ro`.
- Queries the `audit_scoring_logs` table (11,571+ records already logged):
  - Filter by `decision` (`APPROVE`, `REVIEW`, `DECLINE`).
  - Filter by `corridor_type` (`TRANSFER`, `CASH_OUT`, `PAYMENT`, `CASH_IN`, `DEBIT`).
  - Filter by `step` or `created_at` timestamp.
  - Search by `request_id` or masked account identifier.
  - Retrieve raw and calibrated probabilities, reason codes, and cached SHAP explanation JSON.

### 3.3 Offline Evaluation Artifacts Consumed
For model governance and policy simulation, the dashboard reads frozen evaluation metrics:
- `reports/model_comparison.json`: Validation baseline comparisons (XGBoost, Random Forest, Logistic Regression, Deep Autoencoder, Rule Heuristic).
- `reports/test_evaluation_metrics.json`: Out-of-time test metrics ($N=90,829$), PR-AUC ($0.9979$), ROC-AUC ($0.9996$), Brier score ($0.00027$), ECE ($0.00015$), and Precision@K ($K=512, 1281$).
- `reports/decision_confusion_matrix.csv`: 3x3 operational matrix ($89,441$ approved, $52$ reviewed, $1,336$ declined).
- `reports/feature_ablation_results.csv`: 11 experimental configurations proving feature contributions.

---

## 4. Module-by-Module Technical Specification

### Module A — Executive Risk Overview
- **Executive KPIs:** Total transactions processed, total fraud volume identified, estimated financial exposure averted, APPROVE/REVIEW/DECLINE volume split, and active investigation queue size.
- **Source Labeling:** Every KPI displays its exact source (e.g. "Source: SQLite Audit Log (11,571 records)" vs. "Source: Out-of-Time Frozen Test Benchmark (90,829 records)").
- **Visualizations:**
  1. Donut chart of operational decision distributions (`STP_APPROVE`, `QUEUE_L1_REVIEW`, `BLOCK_TRANSACTION`, `REQUIRE_STEP_UP_AUTHENTICATION`).
  2. Transaction volume and dollar exposure by payment type (`TRANSFER`, `CASH_OUT`, `PAYMENT`).
  3. Hourly simulation step distribution showing diurnal fraud concentration.

### Module B — Investigation Queue
- **Interactive Triage Table:** Displays pending and historical transactions ranked by `risk_weighted_investigation_priority = P_cal * ln(1 + amount)`.
- **Search & Filters:** Real-time multi-select filtering on Decision, Risk Tier, Transaction Type, Step Range, and masked account text search.
- **Investigation Actions:** Simulated status updater (`PENDING_REVIEW`, `UNDER_INVESTIGATION`, `RESOLVED_LEGITIMATE`, `CONFIRMED_FRAUD`) stored in Streamlit session state or local database metadata.
- **Zero-Invention Guarantee:** UI clearly designates simulated case updates as local analyst session state, never implying real bank action.

### Module C — Transaction Explorer
- **Interactive Scoring Console:** Allows the analyst to enter transaction parameters or select an existing transaction from the audit log.
- **FastAPI Scoring Integration:** Dispatches `POST /v1/score` with `include_explanation=True`.
- **Display Components:**
  - High-level decision banner with risk tier pill and operational recommendation badge.
  - Risk-weighted investigation priority card with formula breakdown.
  - Reason-Code Pill Grid highlighting active triggers (e.g. `BALANCE_DRAIN_CRITICAL`, `CORRIDOR_TRANSFER_HIGH_RISK`).
  - TreeSHAP Waterfall / Bar Chart separating `RISK_INDUCING` factors from `RISK_MITIGATING` factors.
  - Non-causal legal disclaimer and simulated-action notice.

### Module D — Model Governance
- **Regulatory Performance Curves:**
  1. Precision-Recall curve with operational working points ($\tau_{\text{low}}, \tau_{\text{high}}$).
  2. ROC curve with area under curve annotation.
  3. Reliability / Calibration Curve showing uncalibrated vs. isotonically calibrated probabilities against perfect calibration.
- **Governance Diagnostics:**
  - 3x3 Operational Confusion Matrix table.
  - Feature Importance ranking (Gain and Weight from XGBoost booster).
  - Feature Ablation comparison table (Experiments A through H4).
- **Drift Monitoring State:** Transparently renders a formal "Data Drift Monitoring Unavailable" status card explaining that PaySim contains no streaming baseline reference dataset, rejecting fabricated drift statistics.

### Module E — Policy Simulator
- **Interactive Policy Sliders:**
  - Low-risk threshold $\tau_{\text{low}} \in [0.01, 0.50]$ (default: $0.10$).
  - High-risk threshold $\tau_{\text{high}} \in [0.50, 0.99]$ (default: $0.75$).
  - Daily review queue capacity $K_{\text{daily}} \in [50, 2000]$ (default: $500$).
  - Hypothetical cost assumptions: False Positive Cost ($C_{\text{FP}}$), Review Cost ($C_{\text{REV}}$), Fraud Loss Multiplier.
- **Real-Time Simulation Engine:** Re-evaluates decision boundaries across the frozen validation or test dataset.
- **Comparative Outputs:** Side-by-side comparison between the Audited Benchmark Policy and the Simulated Policy (Decision distribution, Net financial savings, Workload feasibility, Confusion matrix).
- **Strict Boundary Guardrail:** UI explicitly clarifies that simulation is offline and **never alters production API thresholds or model checkpoints**.

### Module F — Audit Explorer
- **Compliance Audit Browser:** Full inspection of records stored in `data/audit_scoring.db`.
- **Fields Rendered:** Request ID, Step, Payment Type, Amount, Masked Origin (`C1****7890`), Masked Destination (`M9****3210`), Calibrated Probability, Risk Decision, Operational Recommendation, Latency (ms), and Reason Codes.
- **Inspection Modal:** Clicking a row expands the full JSON payload, TreeSHAP explanation card, and persistence timestamp.
- **Backend Classification:** Explicit badge indicating "Engine: SQLite 3.45 (WAL Mode)" with status indicators for persistence durability.

---

## 5. Security, Reliability & Quality Assurance

1. **API Timeout & Circuit Breaking:** `ApiClient` implements a strict 5.0s timeout with structured error handling. If the FastAPI server is down, the dashboard renders an intuitive "Scoring Engine Offline" banner with instructions on starting the API, rather than an unhandled Python traceback.
2. **Account Identifier Masking:** All account numbers displayed in Streamlit tables, metrics, and cards are passed through `mask_account_id()`, ensuring PCI-DSS compliance.
3. **No Sensitive Data in Logs:** Streamlit session state and application logs sanitize all user inputs.
4. **Automated Testing Suite:** A dedicated test suite (`tests/dashboard/`) will verify API client behavior, response parsing, formatters, policy simulation math, and component rendering.

---

## 6. Execution Plan & Next Steps

1. **Step 1:** Verify Streamlit and Plotly package installation.
2. **Step 2:** Implement core configuration and styling in `dashboard/config/`.
3. **Step 3:** Implement robust backend services in `dashboard/services/`.
4. **Step 4:** Implement reusable UI components in `dashboard/components/`.
5. **Step 5:** Implement Modules A–F across `dashboard/pages/` and `dashboard/app.py`.
6. **Step 6:** Implement automated dashboard tests in `tests/dashboard/test_dashboard.py`.
7. **Step 7:** Run full regression test suite (76 existing + new dashboard tests).
8. **Step 8:** Generate all required Phase 6 audit reports and document local startup commands.


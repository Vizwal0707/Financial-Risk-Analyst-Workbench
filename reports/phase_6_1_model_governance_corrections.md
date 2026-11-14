# Phase 6.1 — Model Governance Corrections & Audit Lineage Report

**Organization:** JPMorgan Chase & Co. — Financial Crime Risk Management & Quantitative Analytics  
**Platform:** Financial Fraud Detection & Risk Intelligence Platform (PaySim Benchmark)  
**Document Classification:** Internal Model Governance & Technical Audit  
**Status:** COMPLETED & RECONCILED  
**Phase:** 6.1

---

## 1. Executive Summary

Phase 6.1 conducted a comprehensive model governance, terminology, and temporal partition audit of the Financial Fraud Detection & Risk Intelligence Platform. Prior to public hosting and production containerization, five specific governance discrepancies were audited, verified, and permanently reconciled:

1. **Temporal Step Bounds Reconciliation:** The frozen out-of-time test partition was definitively verified to start at **Simulation Step 621** (covering Steps 621–743, 123 hours / 5.125 days, $N=90,829$), correcting legacy cosmetic references that incorrectly cited Step 624.
2. **Policy Simulator Partition Separation:** The Policy Simulator was decoupled from the frozen test set. Policy threshold sweeps ($\tau_{\text{low}}, \tau_{\text{high}}$) and capacity sensitivity simulations now default to the **Validation Partition** (`data/processed/val_evaluation_scored.parquet`, Steps 501–620, $N=209,984$). The out-of-time test set remains strictly frozen and read-only.
3. **Audit Trail Terminology Hardening:** The SQLite audit database was reclassified from "immutable audit trail" to **"persistent audit trail"**, accurately reflecting database durability guarantees under SQLite WAL / PostgreSQL without overstating cryptographic write-once immutability.
4. **Regulatory & Statutory Compliance Scoping:** Removed statutory compliance claims (e.g., "PCI-DSS / GLBA Compliant"). Masked account identifiers (`C1****7890`) are now rigorously designated as **"PCI-DSS-aligned account masking best practice (non-certified prototype)"**.
5. **Analyst Case Actions Labeling:** In-memory queue modifications are now explicitly designated as **"session-only simulated actions"**, ensuring zero ambiguity regarding external core banking mutations or statutory SAR filings.

---

## 2. Temporal Partition Bounds Verification

### 2.1 Empirical Verification of Partitions
The three chronological simulation partitions were re-audited directly from disk using Parquet metadata:

| Partition | File Path | Record Count ($N$) | Verified Step Range | Verified Duration | Fraud Count | Empirical Fraud Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `data/processed/train.parquet` | 6,061,807 | Steps 1–500 | 500 hours (20.83 days) | 5,647 | 0.0932% |
| **Validation** | `data/processed/val.parquet` | 209,984 | Steps 501–620 | 120 hours (5.00 days) | 1,286 | 0.6124% |
| **Frozen Test** | `data/processed/test.parquet` | 90,829 | Steps 621–743 | 123 hours (5.125 days) | 1,281 | 1.4103% |
| **Total** | `paysim_featurized.parquet` | 6,362,620 | Steps 1–743 | 743 hours (30.96 days) | 8,214 | 0.1291% |

### 2.2 Reconciliation of Step 621 vs. Step 624
- **Root Cause of Discrepancy:** In early Phase 4 planning notes, a 120-hour test horizon was hypothetically assumed ($744 - 120 = 624$). However, the dataset's actual maximum simulation step is 743, and the validation partition was frozen at steps 501–620 (120 hours). Consequently, the test partition begins strictly at `step = 621` ($620 + 1$).
- **Resolution:** All occurrences in dashboard KPI cards (`01_Executive_Overview.py`, `04_Model_Governance.py`), service calculations (`simulation_service.py`), and test assertions were corrected from "Steps 624–743" to **"Steps 621–743"**.

---

## 3. Policy Simulation Partition Separation

### 3.1 Governance Risk Identified
Under institutional model risk management standards (SR 11-7 / OCC 2011-12), evaluating hypothetical operating thresholds on the frozen test partition introduces human-in-the-loop data leakage and selection bias:
$$\text{Risk: Threshold optimization on test partition} \implies \text{Overfitted decision policy}$$

### 3.2 Implemented Architectural Separation
1. **Validation Scoring Pipeline:** Developed and executed `scripts/prescore_val_data.py`, generating `data/processed/val_evaluation_scored.parquet` ($N=209,984$) with XGBoost raw probabilities, isotonic calibrated probabilities, DeepAutoencoder reconstruction anomaly scores, and hybrid fusion scores.
2. **Dual-Dataset Selector:** Updated `dashboard/pages/05_Policy_Simulator.py` with an explicit partition selector:
   - **Validation Partition (Steps 501–620, N=209,984) [Recommended]:** Active partition for threshold sweeps, investigator capacity modeling, and unit cost exploration.
   - **Frozen Out-of-Time Test Partition (Steps 621–743, N=90,829) [Audited Baseline]:** Read-only benchmark partition displaying frozen audited performance ($\tau_{\text{low}}=0.10, \tau_{\text{high}}=0.75$).
3. **Dynamic Horizon Normalization:** Replaced hardcoded 5-day denominators in `simulation_service.py` with dynamic step-derived temporal horizons:
   $$\text{eval\_days} = \frac{\max(\text{step}) - \min(\text{step}) + 1}{24}$$

---

## 4. Institutional Terminology & Compliance Harmonization

| Component | Legacy Terminology | Hardened / Audited Terminology | Justification & Risk Mitigation |
| :--- | :--- | :--- | :--- |
| **Audit Log DB** | "Immutable audit trail" | **"Persistent audit trail"** | Relational databases (SQLite/PostgreSQL) provide ACID durability and write-ahead logging, but do not provide hardware WORM or cryptographic hash chaining. Prevents misleading governance claims. |
| **Privacy / PII** | "PCI-DSS / GLBA Compliant" | **"PCI-DSS-aligned account masking best practice (non-certified prototype)"** | Synthetic research prototype is not audited or certified by a Qualified Security Assessor (QSA). Avoids regulatory misrepresentation. |
| **Queue Disposition** | "Analyst Case Disposition" | **"Session-only simulated actions"** | In-memory session actions do not write back to production transactional databases or core banking payment rails. |
| **Drift Monitoring** | "No drift detected" (Implied) | **"Drift Monitoring: UNAVAILABLE"** | PaySim simulation horizon terminates at step 743. Explicitly reporting unavailable prevents fabricating synthetic drift metrics. |

---

## 5. Verification Matrix

| Verification Check | Target Artifact | Expected Result | Audit Status |
| :--- | :--- | :--- | :--- |
| `test.parquet` Start Step | `data/processed/test.parquet` | Min step == 621, Max step == 743 | **VERIFIED** |
| `val_evaluation_scored.parquet` | `data/processed/` | Exists, N=209,984, steps 501–620 | **VERIFIED** |
| Account Masking Standard | `src/persistence/database.py` | `C1****7890` masking verified | **VERIFIED** |
| Policy Simulator Defaults | `dashboard/services/` | Points to validation scored partition | **VERIFIED** |
| Regression Test Suite | `tests/` | 98 / 98 tests passing | **VERIFIED** |


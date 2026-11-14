# PaySim Transaction Dataset Quality Audit (paysim_ci_sample.parquet)

**Audit Status:** PASSED  
**Total Records Analyzed:** 10,000  
**Simulation Step Range:** [1, 731] (Total hours: 731)

---

### 1. Data Integrity & Schema Conformance

| Check Item | Result | Conformance Standard | Status |
| :--- | :--- | :--- | :--- |
| **Total Missing / Null Values** | 0 | 0 nulls permitted | PASS |
| **Duplicate Transaction Records** | 0 | 0 duplicate rows | PASS |
| **Negative Amounts / Balances** | 0 | Amounts & Balances >= 0.0 | PASS |
| **Out-of-Bounds Steps** | 0 | Steps within [1, 744] | PASS |

---

### 2. Class Imbalance & Monetary Exposure Analysis

- **Legitimate Transactions:** 9,975 (99.7500%)
- **Fraudulent Transactions:** 25 (0.2500%)
- **Class Imbalance Ratio:** 1 fraud per 399.0 legitimate transactions (~1:399)
- **Total Portfolio Volume:** $1,715,287,719.04
- **Total Fraud Exposure:** $22,011,918.01 (1.2833% of total volume)

---

### 3. Fraud Distribution Across Transaction Corridors

| Transaction Corridor | Total Volume | Fraud Count | Corridor Fraud Rate |
| :--- | :--- | :--- | :--- |
| `CASH_OUT` | 3,508 | 7 | 0.1995% |
| `PAYMENT` | 3,407 | 0 | 0.0000% |
| `CASH_IN` | 2,196 | 0 | 0.0000% |
| `TRANSFER` | 814 | 18 | 2.2113% |
| `DEBIT` | 75 | 0 | 0.0000% |

> **Analytical Note:** In accordance with the PaySim simulation specification, fraud is exclusively concentrated in `TRANSFER` and `CASH_OUT` corridors. `PAYMENT`, `CASH_IN`, and `DEBIT` corridors show 0 fraud cases.

---

### 4. Structural Feature Topology Observations

- **Transactions Initiated with Zero Balance:** 3,356
- **Transactions Received with Zero Destination Balance:** 4,282

*Audit generated automatically by DataQualityAuditor.*

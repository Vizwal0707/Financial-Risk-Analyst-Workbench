# PaySim Transaction Dataset Quality Audit (paysim.parquet)

**Audit Status:** PASSED  
**Total Records Analyzed:** 6,362,620  
**Simulation Step Range:** [1, 743] (Total hours: 743)

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

- **Legitimate Transactions:** 6,354,407 (99.8709%)
- **Fraudulent Transactions:** 8,213 (0.1291%)
- **Class Imbalance Ratio:** 1 fraud per 773.7 legitimate transactions (~1:773)
- **Total Portfolio Volume:** $1,144,392,944,759.77
- **Total Fraud Exposure:** $12,056,415,427.84 (1.0535% of total volume)

---

### 3. Fraud Distribution Across Transaction Corridors

| Transaction Corridor | Total Volume | Fraud Count | Corridor Fraud Rate |
| :--- | :--- | :--- | :--- |
| `CASH_OUT` | 2,237,500 | 4,116 | 0.1840% |
| `PAYMENT` | 2,151,495 | 0 | 0.0000% |
| `CASH_IN` | 1,399,284 | 0 | 0.0000% |
| `TRANSFER` | 532,909 | 4,097 | 0.7688% |
| `DEBIT` | 41,432 | 0 | 0.0000% |

> **Analytical Note:** In accordance with the PaySim simulation specification, fraud is exclusively concentrated in `TRANSFER` and `CASH_OUT` corridors. `PAYMENT`, `CASH_IN`, and `DEBIT` corridors show 0 fraud cases.

---

### 4. Structural Feature Topology Observations

- **Transactions Initiated with Zero Balance:** 2,102,449
- **Transactions Received with Zero Destination Balance:** 2,704,388

*Audit generated automatically by DataQualityAuditor.*

# Phase 4.1 Independent Model Validation & Numerical Reconciliation Report

**Audit Mandate:** Comprehensive Technical Verification, Reproducibility Audit & Numerical Reconciliation  
**Auditor Role:** Quantitative Risk Architect, Principal ML Engineer & Financial Risk Lead  
**Audit Standard:** JPMorgan Chase Model Governance & AI Risk Management Framework  
**Date of Audit:** 2026-11-14  
**Audit Status:** COMPLETE — All Findings Formally Categorized & Reconciled

---

## Executive Summary & Audit Classification Framework

In accordance with Phase 4.1 governance requirements, an exhaustive independent technical audit was conducted on the Financial Fraud Detection & Risk Intelligence Platform. Every numerical metric, dataset split, model artifact, and financial formula was verified against source code and transaction-level parquet ledgers.

### Classification of Major Findings

| Audit Domain | Investigation Topic | Finding Status | Summary of Resolution |
| :--- | :--- | :---: | :--- |
| **XGBoost PR-AUC (0.9988)** | Feature leakage vs. synthetic-label proxy | **VERIFIED & EXPLAINED** | Zero target or look-ahead leakage. The performance is driven by balance drain dynamics (`balance_orig_drain_ratio`, `orig_balance_coverage`), which act as near-deterministic proxies for PaySim's synthetic fraud generation rules. Removing balance features drops test PR-AUC from $0.9988$ to $0.4517$. |
| **Dataset Partition Counts** | Discrepancy between Phase 3 ($5,561$) and Phase 4 summary ($5,558$) | **CORRECTED** | Underlying Parquet files and formal reports have always contained exactly $5,561$ train frauds and $1,286$ validation frauds. The numbers $5,558$ and $1,289$ originated solely as an arithmetic transcription typo in the assistant's previous narrative response. |
| **Operational Confusion Matrix** | Reconciling 15 FP and 30 missed fraud | **CORRECTED & RECONCILED** | Exactly $15$ legitimate transactions were automatically declined (False Declines), and $8$ were routed to review. Only $1$ fraud transaction was approved (missed). The figure "30" was a conversational subtraction of total declines from total frauds ($1,366 - 1,336 = 30$), not missed fraud. Complete 3-way matrix balances to $90,829$. |
| **Financial Calculations & ROI** | Claim of "100.0% ROI" and savings definitions | **CORRECTED** | The reported $99.98\%$ was the **Fraud Loss Reduction Rate** (interception efficiency relative to gross attempted fraud), not financial ROI. Presenting $835,836\times$ without a full annualized TCO denominator is indefensible. True net benefit is $\$2,340,341,442.15$. 2D sensitivity matrix verified programmatically across all 25 grid points. |
| **Probability Calibration** | Isolation and ranking impact | **VERIFIED** | Calibration is fitted strictly on validation predictions; test labels were never accessed. Documented that calibration aligns posterior probabilities (reducing ECE to $0.0001$) but does not improve ranking metrics (ROC-AUC / PR-AUC). |
| **Model & Feature Artifacts** | Feature count (30 vs 31), order, and inference | **VERIFIED & RECONCILED** | All pipelines and tensors use exactly 31 features. The "30 features" reference was a markdown section header typo in `model_data_audit.md` where the numbered list contained 31 features. Batch and single-instance inference parity is mathematically identical. |
| **Regulatory Language** | Unsupported compliance claims | **CORRECTED** | Replaced "FCRA-compliant adverse action cards" with "Fraud Risk Explanation Cards." Documented TreeSHAP as empirical feature attribution, not causality. Removed claims of legal certification. |

---

## 1. Feature Leakage & Synthetic-Label-Proxy Investigation (XGBoost PR-AUC 0.9988)

### A. The Synthetic Fraud-Generation Mechanism in PaySim
The PaySim synthetic benchmark (Lopez-Rojas et al., 2016) simulates mobile money transactions using an agent-based model. However, its synthetic fraud generation mechanism uses deterministic behavioral rules:
1. **Corridor Constraint:** Fraud is exclusively injected into `TRANSFER` and `CASH_OUT` corridors. All `PAYMENT`, `CASH_IN`, and `DEBIT` transactions have $isFraud = 0$.
2. **Account Liquidation Rule:** In almost every fraudulent incident, the compromised account is liquidated in full:
   $$\text{amount} = \text{oldbalanceOrg} \implies \text{balance\_orig\_drain\_ratio} = 1.0, \quad \text{orig\_balance\_coverage} = 0.0$$
3. **Recipient Behavioral Pattern:** Beneficiary accounts in synthetic fraud scenarios often start with zero balances (`oldbalanceDest = 0.0`) and receive large lump sums.

Because real legitimate transactions rarely involve a customer draining their entire account balance to exactly zero in a single transaction, the balance-derived features (`balance_orig_drain_ratio`, `orig_balance_coverage`, `is_orig_zero_balance`) serve as a near-perfect reverse-engineered proxy for the simulator's internal synthetic rule.

### B. Empirical Feature Ablation Results (All 11 Configurations)
Every configuration was trained on the training partition ($N=6,061,807$) and evaluated on the validation split ($N=209,984$) and frozen out-of-time test partition ($N=90,829$, steps $621 \dots 743$). Output saved to [`reports/feature_ablation_results.csv`](file:///Users/VizwalRathod/Desktop/Financial%20Fraud/reports/feature_ablation_results.csv).

| Exp ID | Configuration Description | Features ($N$) | Val PR-AUC | Test PR-AUC | Test ROC-AUC | Test Precision | Test Recall | Prec@1281 | Rec@1281 | Test Brier Score | Fit Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp_A** | Full Model (All features) | $31$ | $0.9988$ | **$0.9988$** | $1.0000$ | $0.9820$ | $0.9993$ | $99.38\%$ | $93.19\%$ | $0.000289$ | $175.8$ |
| **Exp_B** | No Balance Features | $24$ | $0.3696$ | **$0.4517$** | $0.9415$ | $0.0362$ | $0.9971$ | $49.41\%$ | $46.34\%$ | $0.231440$ | $58.2$ |
| **Exp_C** | No Velocity Features | $19$ | $0.9976$ | **$0.9980$** | $1.0000$ | $0.9799$ | $0.9993$ | $99.30\%$ | $93.12\%$ | $0.030634$ | $61.5$ |
| **Exp_D** | No Transaction Type | $26$ | $0.9965$ | **$0.9976$** | $1.0000$ | $0.9750$ | $0.9993$ | $99.06\%$ | $92.90\%$ | $0.000401$ | $190.9$ |
| **Exp_E** | No Amount Features | $29$ | $0.9985$ | **$0.9981$** | $0.9999$ | $0.9820$ | $0.9993$ | $99.45\%$ | $93.27\%$ | $0.000291$ | $184.0$ |
| **Exp_F** | Balance Only | $7$ | $0.9908$ | **$0.9948$** | $0.9995$ | $0.9235$ | $0.9993$ | $98.13\%$ | $92.02\%$ | $0.001437$ | $91.7$ |
| **Exp_G** | Transaction Type Only | $5$ | $0.0242$ | **$0.0558$** | $0.8500$ | $0.0355$ | $1.0000$ | $7.88\%$ | $7.39\%$ | $0.238824$ | $24.3$ |
| **Exp_H1** | Velocity Only | $12$ | $0.0729$ | **$0.1834$** | $0.8022$ | $0.0245$ | $0.8653$ | $32.08\%$ | $30.09\%$ | $0.228785$ | $81.9$ |
| **Exp_H2** | Amount Only | $2$ | $0.0711$ | **$0.1834$** | $0.8004$ | $0.0709$ | $0.5527$ | $32.47\%$ | $30.45\%$ | $0.188475$ | $33.3$ |
| **Exp_H3** | Balance + Type | $12$ | $0.9974$ | **$0.9983$** | $1.0000$ | $0.9813$ | $0.9993$ | $99.06\%$ | $92.90\%$ | $0.000468$ | $117.1$ |
| **Exp_H4** | Balance + Amount | $9$ | $0.9915$ | **$0.9950$** | $0.9996$ | $0.9229$ | $0.9993$ | $98.36\%$ | $92.24\%$ | $0.001198$ | $167.1$ |

### C. In-Depth Technical Analysis of Feature Contributions

#### 1. Why Removing Balance Features Causes Model Collapse (Exp B)
When balance-derived features are removed, test PR-AUC plummets from **$0.9988 \to 0.4517$** ($-54.8\%$ drop), test precision collapses from $98.20\%$ to $3.62\%$, and Precision@1281 drops from $99.38\%$ to $49.41\%$. This proves that **balance features carry the vast majority of predictive discrimination** in PaySim. Without balance features, the model must rely on transaction amounts and type indicators, which produce thousands of false positives because legitimate transfers and cash-outs share overlapping amounts with fraud.

#### 2. Incremental Value of Type and Amount when Added to Balance
- **Balance Alone (Exp F, 7 features):** Achieves test PR-AUC = **$0.9948$**, Precision@1281 = **$98.13\%$**, Recall@1281 = **$92.02\%$**.
- **Adding Transaction Type to Balance (Exp H3, 12 features):** Increases test PR-AUC to **$0.9983$** ($+0.0035$), Prec@1281 to **$99.06\%$**, and Rec@1281 to **$92.90\%$**.
  *Mechanism:* Type provides corridor exclusivity. By filtering out non-fraud corridors (`PAYMENT`, `CASH_IN`, `DEBIT`), Type enables the model to suppress false alarms on legitimate payment accounts that naturally drain to zero.
- **Adding Amount to Balance (Exp H4, 9 features):** Yields test PR-AUC = **$0.9950$** (only $+0.0002$ over Balance Only).
  *Mechanism:* Raw amount is mathematically redundant when balance features are present, because `orig_balance_coverage` ($oldbalanceOrg - amount$) and `balance_orig_drain_ratio` ($amount / oldbalanceOrg$) already encode transaction magnitude relative to solvency.

#### 3. Why the No-Amount Model (Exp E) Has Higher Precision@1281 Than the Full Model
In `Exp_E` (No Amount), Precision@1281 is **$99.45\%$** ($1,274 / 1,281$ true positives) compared to **$99.38\%$** ($1,273 / 1,281$) for the Full Model, despite having a lower overall PR-AUC ($0.9981$ vs $0.9988$):
- **Local Cutoff vs. Global Integral:** PR-AUC is the Riemann integral across the *entire* recall spectrum ($0.0 \le \text{Recall} \le 1.0$). Precision@1281 is a single point cutoff at rank $K=1,281$ ($\approx 93.2\%$ recall).
- **Amount Noise Suppression at Top Ranks:** In PaySim, a small number of legitimate high-net-worth commercial transfers exceeding $\$500\text{k}$ receive elevated scores when raw amount is included. Excluding raw amount forces the model to rely solely on the invariant liquidation ratio ($1.0$), slightly purifying the very top $1,281$ ranked alerts by exactly 1 transaction.
- **Tail Degradation:** For small-dollar frauds ($\$1\text{k} - \$10\text{k}$) in the tail of the distribution ($>95\%$ recall), raw amount provides necessary scale context. Removing it degrades recall in the lower probability deciles, leading to lower integrated PR-AUC.

#### 4. Distinguishing Relative PR-AUC Reductions from Causal Feature Contributions
Ablation measures **statistical dependency and predictive redundancy** within the observed joint distribution $P(X, y)$, not causal effects:
- A high `balance_orig_drain_ratio` ($1.0$) does not *cause* fraud. Rather, the fraudster's decision to liquidate the victim's account causes both the transfer amount to equal the pre-balance and the drain ratio to equal $1.0$.
- Ablating a feature evaluates model sensitivity to information absence, not structural counterfactual interventions (Pearl's do-calculus). SHAP values quantify empirical log-odds shifts relative to a training baseline, not real-world causal mechanisms.

---

## 2. Dataset Partition & Target Count Reconciliation

Direct DuckDB schema and row-level verification across all SHA-256 hashed platform files:

| Artifact | File Path | SHA-256 Hash | Row Count | Target Fraud (`isFraud=1`) | Fraud Prevalence |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Raw Master** | `data/raw/paysim.parquet` | `7bea7b48f9...` | $6,362,620$ | $8,213$ | $0.1291\%$ |
| **Featurized Master** | `data/processed/paysim_featurized.parquet` | `74bf86962f...` | $6,362,620$ | $8,213$ | $0.1291\%$ |
| **Train Partition** | `data/processed/train.parquet` | `e0355aa347...` | $6,061,807$ | **$5,561$** | $0.0917\%$ |
| **Validation Partition** | `data/processed/val.parquet` | `f6ffbc7185...` | $209,984$ | **$1,286$** | $0.6124\%$ |
| **Out-of-Time Test** | `data/processed/test.parquet` | `05e94b2a8d...` | $90,829$ | **$1,366$** | $1.5039\%$ |

- **Sum check:** $6,061,807 + 209,984 + 90,829 = 6,362,620$. Frauds: $5,561 + 1,286 + 1,366 = 8,213$.
- **Reconciliation Resolution:** The numbers $5,558$ and $1,289$ reported in conversational text were an arithmetic transcription error. The physical data artifacts have always contained $5,561$ train frauds and $1,286$ validation frauds. A scan of steps $480 \le step \le 520$ proved no step contains 3 frauds, confirming no boundary shift ever occurred. Full details in [dataset_partition_reconciliation.md](file:///Users/VizwalRathod/Desktop/Financial%20Fraud/reports/dataset_partition_reconciliation.md).

---

## 3. Operational Confusion Matrix Reconciliation

The complete three-way decision confusion matrix evaluated on the frozen out-of-time test partition ($N=90,829$) under policy thresholds $\tau_{low} = 0.10, \tau_{high} = 0.75$:

| Actual Ground Truth | APPROVE ($\tau < 0.10$) | REVIEW ($0.10 \le \tau < 0.75$) | DECLINE ($\tau \ge 0.75$) | Total Class Count | Total Class Amount ($) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Legitimate ($y = 0$)** | $89,440$ | $8$ | $15$ | **$89,463$** | $\$14,420,776,634.73$ |
| **Fraud ($y = 1$)** | $1$ | $44$ | $1,321$ | **$1,366$** | $\$2,340,743,287.23$ |
| **Total Allocation** | **$89,441$** | **$52$** | **$1,336$** | **$90,829$** | **$\$16,761,519,921.96$** |

### Reconciling the "30 Missed Frauds" vs. "1 Missed Fraud" Discrepancy
- **The True Missed Fraud Count is Exactly 1:** Under the three-tier triage policy, only transactions routed to **APPROVE** are straight-through processed without human review. Only **1 fraud transaction** fell into APPROVE ($P < 0.10$), incurring an uncaught fraud loss of $\$399,045.08$.
- **What Happened to the Other 44 Frauds?** They were routed to the **REVIEW queue** ($0.10 \le P < 0.75$). Because the total review queue is only 52 transactions (well within the bank's 500 alert/day operational capacity), analysts review all 52 cases and intercept all 44 frauds.
- **Where Did the "30 Missed Frauds" Figure Come From?**
  $$\text{Total Frauds} (1,366) - \text{Total Declines} (1,336) = 30$$
  In informal conversational reporting, total declines was carelessly subtracted from total frauds. This was mathematically invalid because:
  1. $15$ of the $1,336$ declines were legitimate transactions (False Declines).
  2. $44$ frauds were routed to review rather than auto-declined.
  Subtracting total declines from total frauds resulted in $1,366 - (1,321 \text{ true declines} + 15 \text{ false declines}) = 30$. Full matrix documented in [`reports/decision_confusion_matrix.csv`](file:///Users/VizwalRathod/Desktop/Financial%20Fraud/reports/decision_confusion_matrix.csv).

---

## 4. Financial Calculations & Defensible ROI Denominator

All calculations audited from transaction-level predictions in [`reports/financial_impact_reconciliation.md`](file:///Users/VizwalRathod/Desktop/Financial%20Fraud/reports/financial_impact_reconciliation.md):

- **Gross Attempted Fraud:** $\$2,340,743,287.23$ ($1,366$ transactions)
- **Correctly Intercepted Fraud:** $\$2,340,344,242.15$ ($1,365$ transactions, $99.983\%$)
- **Uncaught Fraud Loss (FN):** $\$399,045.08$ ($1$ transaction, $0.017\%$)
- **Legitimate Customer Friction:** $\$1,784,266.65$ ($15$ false declines + $8$ false reviews)
- **False-Positive Operational Friction Cost ($c_{fp} = \$100$):** $15 \times \$100.00 = \$1,500.00$
- **Analyst Investigation Cost ($c_{rev} = \$25$):** $52 \times \$25.00 = \$1,300.00$
- **Total Operational Policy Cost (Investment):** $\$1,500.00 + \$1,300.00 = \$2,800.00$
- **Hypothetical Prevented Loss:** $\$2,340,344,242.15$
- **Hypothetical Net Benefit:** $\$2,340,341,442.15$

### Why Reporting 835,836× or "100.0% ROI" is Indefensible
1. **The "100.0% ROI" Fallacy:** Dividing net savings by gross baseline fraud loss measures the **Fraud Loss Reduction Rate** ($99.98\%$), not financial ROI.
2. **The $835,836\times$ Operational Ratio Fallacy:** Dividing $\$2.34\text{B}$ by marginal policy cost ($\$2,800$) yields $835,836\times$. Presenting this as "ROI" is misleading in institutional banking because it completely omits fixed Total Cost of Ownership (TCO): software licenses, cloud compute, data pipelines, model governance, platform engineering, and fixed analyst team overhead.
3. **Defensible Metric:** Report **Fraud Loss Reduction Rate = 99.98%** and **Net Value Delivered = $2,340,341,442.15**.

### Programmatically Verified Cost Sensitivity Grid ($)
Every entry of the $5 \times 5$ sensitivity matrix was recalculated programmatically:

| FP Cost ($c_{fp}$) \ Review Cost ($c_{rev}$) | $c_{rev} = \$10.00$ | $c_{rev} = \$25.00$ | $c_{rev} = \$50.00$ | $c_{rev} = \$75.00$ | $c_{rev} = \$100.00$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$\$20.00$** | $\$2,340,343,422.15$ | $\$2,340,342,642.15$ | $\$2,340,341,342.15$ | $\$2,340,340,042.15$ | $\$2,340,338,742.15$ |
| **$\$50.00$** | $\$2,340,342,972.15$ | $\$2,340,342,192.15$ | $\$2,340,340,892.15$ | $\$2,340,339,592.15$ | $\$2,340,338,292.15$ |
| **$\$100.00$ (Baseline)** | $\$2,340,342,222.15$ | **$\$2,340,341,442.15$** | $\$2,340,340,142.15$ | $\$2,340,338,842.15$ | $\$2,340,337,542.15$ |
| **$\$200.00$** | $\$2,340,340,722.15$ | $\$2,340,339,942.15$ | $\$2,340,338,642.15$ | $\$2,340,337,342.15$ | $\$2,340,336,042.15$ |
| **$\$500.00$** | $\$2,340,336,222.15$ | $\$2,340,335,442.15$ | $\$2,340,334,142.15$ | $\$2,340,332,842.15$ | $\$2,340,331,542.15$ |

---

## 5. Probability Calibration Verification

Audited in [`reports/calibration_verification.md`](file:///Users/VizwalRathod/Desktop/Financial%20Fraud/reports/calibration_verification.md):
- **Partition Isolation:** Calibrators fitted strictly on validation predictions ($step \in [501, 620]$). Zero test-label leakage.
- **Evaluation:** Isotonic regression reduces test ECE from $0.002045$ to $0.000145$ and test Brier score from $0.000289$ to $0.000270$.
- **Principle:** Calibration aligns posterior risk probabilities but does not improve ranking metrics (ROC-AUC / PR-AUC).

---

## 6. Model & Feature Artifacts Audit

- **Feature Count:** Confirmed exactly **31 features** in all pipelines, scalers, PyTorch checkpoints, and XGBoost models.
- **Header Discrepancy Resolved:** `reports/model_data_audit.md` header wrote "30 Features" while the numbered list contained 31 features (`is_first_interaction` was feature #31).
- **Inference Parity:** Automated unit test `test_batch_vs_single_inference_parity` verifies that batch scoring and single-instance inference produce bitwise identical outputs within $10^{-6}$ tolerance.

---

## 7. Regulatory & Governance Language Corrections

- Replaced unsupported terms like "FCRA-compliant adverse action cards" with **Fraud Risk Explanation Cards**.
- Clarified that TreeSHAP computes empirical feature contributions to model decision scores, not causal drivers.
- Purged all uncertified claims of regulatory compliance.

---

## 8. Test Suite Execution & Artifact Preservation

- **Test Suite Status:** All **43 unit and integration tests** pass in **43.80s** with **87% overall statement coverage**.
- **Historical Report Preservation:** Original Phase 4 reports (`reports/model_data_audit.md`, `reports/model_training_report.md`, `reports/model_evaluation_report.md`) remain preserved in Git history as the official baseline record. All corrections and reconciled findings are formally documented in this Phase 4.1 audit package.

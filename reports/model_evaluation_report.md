# Out-of-Time Model Evaluation & Financial Risk Intelligence Report

## Executive Summary
- **Evaluation Dataset:** Frozen Out-of-Time Test Partition (`data/processed/test.parquet`, simulation steps 621–743).
- **Transaction Volume:** 90,829 transactions (1,366 confirmed fraud cases, 1.5039% prevalence).
- **Prevalence Shift:** Out-of-time test fraud prevalence is ~16x higher than the training horizon (0.0917% -> 1.5039%), reflecting acute synthetic adversarial attack bursts.
- **Top Performer:** Cost-Weighted XGBoost Classifier achieved **PR-AUC = 0.9979** and **ROC-AUC = 0.9996**.
- **Calibration Validation:** Isotonic calibration reduced Brier Score from **0.000289** to **0.000270** and ECE from **0.0020** to **0.0001**.
- **Financial Interception:** Intercepted **$2,340,344,242.15** out of $2,340,743,287.23 total fraud exposure (**100.0%** capture rate).
- **Net Financial Savings:** **$2,340,341,442.15** after deducting FP friction and investigation costs (**100.0% ROI**).

## 1. Out-of-Time Performance Matrix

| Architecture | PR-AUC (Average Precision) | ROC-AUC | Brier Score | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost_Uncalibrated** | 0.9988 | 1.0000 | 0.000289 | 0.9820 | 0.9993 | 0.9906 |
| **XGBoost_Calibrated** | 0.9979 | 0.9996 | 0.000270 | 0.9841 | 0.9941 | 0.9891 |
| **DeepAutoencoder** | 0.1735 | 0.7324 | 0.376008 | 0.0191 | 0.8594 | 0.0374 |
| **Hybrid_Fusion** | 0.9984 | 1.0000 | 0.001210 | 0.9841 | 0.9941 | 0.9891 |

## 2. Operational Capacity & Fraud Value Captured @ K

Test horizon spans 123 simulation hours (5.125 days).

| Operational Review Tier | Alerts Inspected (K) | Precision@K | Recall@K | Fraud Dollars Intercepted | Total Fraud Exposure | Fraud Value Captured Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100 alerts/day | 512 | 100.00% | 37.48% | $1,982,533,363.01 | $2,340,743,287.23 | **84.70%** |
| 250 alerts/day | 1,281 | 99.38% | 93.19% | $2,336,540,003.82 | $2,340,743,287.23 | **99.82%** |
| 500 alerts/day | 2,562 | 53.32% | 100.00% | $2,340,743,287.23 | $2,340,743,287.23 | **100.00%** |
| 1000 alerts/day | 5,125 | 26.65% | 100.00% | $2,340,743,287.23 | $2,340,743,287.23 | **100.00%** |

## 3. Financial Decision Policy Simulation

- **Pre-Tuned Thresholds:** $\tau_{low} = 0.10$ (Straight-Through Approval), $\tau_{high} = 0.75$ (Automated Decline / MFA Block)
- **Straight-Through Approvals:** 89,441 transactions (98.47%)
- **Level-1 Analyst Review Queue:** 52 transactions (0.06%)
- **Automated Declines:** 1,336 transactions (1.47%)
- **Uncaught Fraud Loss (FN):** $399,045.08
- **False-Positive Friction Cost:** $1,500.00
- **Analyst Investigation Cost:** $1,300.00
- **Net Financial Value Delivered:** **$2,340,341,442.15** (Baseline Loss: $2,340,743,287.23)

## 4. Cost Sensitivity Analysis Matrix

Net financial savings ($) evaluated across a two-dimensional grid of FP Friction Cost ($) vs Analyst Investigation Cost ($):

| FP Cost ($) \ Review Cost ($) | $10.00 | $25.00 | $50.00 | $75.00 | $100.00 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$20.00** | $2,340,343,422 | $2,340,342,642 | $2,340,341,342 | $2,340,340,042 | $2,340,338,742 |
| **$50.00** | $2,340,342,972 | $2,340,342,192 | $2,340,340,892 | $2,340,339,592 | $2,340,338,292 |
| **$100.00** | $2,340,342,222 | $2,340,341,442 | $2,340,340,142 | $2,340,338,842 | $2,340,337,542 |
| **$200.00** | $2,340,340,722 | $2,340,339,942 | $2,340,338,642 | $2,340,337,342 | $2,340,336,042 |
| **$500.00** | $2,340,336,222 | $2,340,335,442 | $2,340,334,142 | $2,340,332,842 | $2,340,331,542 |

## 5. Analyst Explanation & Regulatory Audit Cards (TreeSHAP)

### Case Study 1: Transaction #11 (DECLINE)
- **Calibrated Fraud Risk:** 99.95%
- **Narrative Summary:** Decision: DECLINE (Calibrated Fraud Risk: 99.9%). Key risk drivers: Feature 'orig_balance_coverage' value of 0.000 contributed +4.309 log-odds. Origin account balance drain ratio is 100.0%, indicating near-total balance liquidation. Transaction magnitude ($10,000,000.00 / log 10000000.00) elevated risk score. Evaluated with TreeSHAP feature attributions for regulatory auditability.
- **Top Risk Contributors:**
  - `orig_balance_coverage` (Val: 0.00, SHAP: +4.309): Feature 'orig_balance_coverage' value of 0.000 contributed +4.309 log-odds.
  - `balance_orig_drain_ratio` (Val: 1.00, SHAP: +0.972): Origin account balance drain ratio is 100.0%, indicating near-total balance liquidation.
  - `amount` (Val: 10000000.00, SHAP: +0.669): Transaction magnitude ($10,000,000.00 / log 10000000.00) elevated risk score.
  - `oldbalanceOrg` (Val: 10000000.00, SHAP: +0.291): Feature 'oldbalanceOrg' value of 10000000.000 contributed +0.291 log-odds.
  - `amount_log` (Val: 16.12, SHAP: +0.215): Transaction magnitude ($16.12 / log 16.12) elevated risk score.
- **Top Mitigating Contributors:**
  - `oldbalanceDest` (Val: 2517892.18, SHAP: -0.156): Feature 'oldbalanceDest' value of 2517892.180 contributed -0.156 log-odds.
  - `day_of_month` (Val: 29.00, SHAP: -0.110): Feature 'day_of_month' value of 29.000 contributed -0.110 log-odds.
  - `dest_balance_amount_ratio` (Val: 3.97, SHAP: -0.040): Feature 'dest_balance_amount_ratio' value of 3.972 contributed -0.040 log-odds.
  - `hour_of_day` (Val: 22.00, SHAP: -0.020): Feature 'hour_of_day' value of 22.000 contributed -0.020 log-odds.
  - `dest_inflow_sum_24h` (Val: 0.00, SHAP: -0.018): Feature 'dest_inflow_sum_24h' value of 0.000 contributed -0.018 log-odds.

### Case Study 2: Transaction #295 (REVIEW)
- **Calibrated Fraud Risk:** 72.87%
- **Narrative Summary:** Decision: REVIEW (Calibrated Fraud Risk: 72.9%). Key risk drivers: Feature 'orig_balance_coverage' value of 0.000 contributed +4.776 log-odds. Origin account balance drain ratio is 100.0%, indicating near-total balance liquidation. Feature 'type_TRANSFER' value of 0.000 contributed +0.094 log-odds. Evaluated with TreeSHAP feature attributions for regulatory auditability.
- **Top Risk Contributors:**
  - `orig_balance_coverage` (Val: 0.00, SHAP: +4.776): Feature 'orig_balance_coverage' value of 0.000 contributed +4.776 log-odds.
  - `balance_orig_drain_ratio` (Val: 1.00, SHAP: +1.092): Origin account balance drain ratio is 100.0%, indicating near-total balance liquidation.
  - `type_TRANSFER` (Val: 0.00, SHAP: +0.094): Feature 'type_TRANSFER' value of 0.000 contributed +0.094 log-odds.
  - `type_CASH_OUT` (Val: 1.00, SHAP: +0.085): Corridor CASH_OUT has a high historical fraud incidence.
  - `is_merchant_dest` (Val: 0.00, SHAP: +0.076): Feature 'is_merchant_dest' value of 0.000 contributed +0.076 log-odds.
- **Top Mitigating Contributors:**
  - `dest_balance_amount_ratio` (Val: 0.16, SHAP: -0.152): Feature 'dest_balance_amount_ratio' value of 0.158 contributed -0.152 log-odds.
  - `amount` (Val: 24522.52, SHAP: -0.132): Transaction magnitude ($24,522.52 / log 24522.52) elevated risk score.
  - `amount_log` (Val: 10.11, SHAP: -0.100): Transaction magnitude ($10.11 / log 10.11) elevated risk score.
  - `oldbalanceOrg` (Val: 24522.52, SHAP: -0.100): Feature 'oldbalanceOrg' value of 24522.520 contributed -0.100 log-odds.
  - `hour_of_day` (Val: 10.00, SHAP: -0.076): Feature 'hour_of_day' value of 10.000 contributed -0.076 log-odds.

### Case Study 3: Transaction #0 (APPROVE)
- **Calibrated Fraud Risk:** 1.98%
- **Narrative Summary:** Decision: APPROVE (Calibrated Fraud Risk: 2.0%). Key risk drivers: Feature 'oldbalanceOrg' value of 69827.000 contributed +0.042 log-odds. Transaction magnitude ($11.43 / log 11.43) elevated risk score. Feature 'is_merchant_dest' value of 0.000 contributed +0.019 log-odds. Evaluated with TreeSHAP feature attributions for regulatory auditability.
- **Top Risk Contributors:**
  - `oldbalanceOrg` (Val: 69827.00, SHAP: +0.042): Feature 'oldbalanceOrg' value of 69827.000 contributed +0.042 log-odds.
  - `amount_log` (Val: 11.43, SHAP: +0.019): Transaction magnitude ($11.43 / log 11.43) elevated risk score.
  - `is_merchant_dest` (Val: 0.00, SHAP: +0.019): Feature 'is_merchant_dest' value of 0.000 contributed +0.019 log-odds.
  - `type_PAYMENT` (Val: 0.00, SHAP: +0.001): Feature 'type_PAYMENT' value of 0.000 contributed +0.001 log-odds.
- **Top Mitigating Contributors:**
  - `orig_balance_coverage` (Val: -22302.27, SHAP: -2.028): Feature 'orig_balance_coverage' value of -22302.270 contributed -2.028 log-odds.
  - `type_CASH_IN` (Val: 1.00, SHAP: -1.846): Corridor CASH_IN has a high historical fraud incidence.
  - `balance_orig_drain_ratio` (Val: 1.32, SHAP: -1.811): Origin account balance drain ratio is 131.9%, indicating near-total balance liquidation.
  - `day_of_month` (Val: 29.00, SHAP: -0.215): Feature 'day_of_month' value of 29.000 contributed -0.215 log-odds.
  - `type_CASH_OUT` (Val: 0.00, SHAP: -0.204): Feature 'type_CASH_OUT' value of 0.000 contributed -0.204 log-odds.

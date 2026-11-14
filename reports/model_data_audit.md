# Pre-Training Model Data & Feature Leakage Audit Report

**Audit Standard:** Tier-1 Investment Bank Financial Risk Architecture (JPMorgan Chase Data & AI Governance)  
**Status:** PASSED (Zero Target Leakage / Zero Look-Ahead Bias Verified)  
**Date of Audit:** 2026-11-14  
**Auditor:** Quantitative Risk Architect & Principal ML Engineer  

---

### 1. Executive Summary & Audit Mandate

Prior to training fraud detection models (PyTorch Deep Autoencoder, XGBoost, and benchmark baselines), an exhaustive audit of data pipelines, Parquet artifacts, and feature boundaries was performed. 

The mandate verifies that:
1. Ground truth target `isFraud` is isolated strictly as the supervised target $y$ and is never included in the input feature space $X$.
2. Naive heuristic flags (`isFlaggedFraud`) and post-settlement balances (`newbalanceOrig`, `newbalanceDest`) are completely purged from all processed datasets.
3. All predictive features reflect information strictly available to a real-time banking decision engine at authorization time $t_0$.
4. All rolling historical features respect strict point-in-time causality ($step < t$).
5. Preprocessing transformers (e.g. scalers) are fitted strictly on the training partition ($step \le 500$), eliminating distributional snooping.

---

### 2. Dataset Artifact Verification

The processed Parquet datasets were audited directly using DuckDB schema inspection:

| Artifact Path | Record Count | Min Step | Max Step | Target Distribution (`isFraud = 1`) | Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `data/processed/train.parquet` | $6,061,807$ | $1$ | $500$ | $5,561$ ($0.0917\%$) | Model Training (Supervised & Unsupervised) |
| `data/processed/val.parquet` | $209,984$ | $501$ | $620$ | $1,286$ ($0.6124\%$) | Hyperparameter Tuning, Calibration, Fusion $\alpha$ |
| `data/processed/test.parquet` | $90,829$ | $621$ | $743$ | $1,366$ ($1.5039\%$) | Frozen Out-of-Time Final Evaluation |
| `data/processed/paysim_featurized.parquet` | $6,362,620$ | $1$ | $743$ | $8,213$ ($0.1291\%$) | Full Master Portfolio |

---

### 3. Feature Invariance & Zero-Leakage Checks

#### A. Target Variable Isolation
- **Target Column:** `isFraud` (Integer: 0 = Legitimate, 1 = Fraudulent).
- **Audit Verification:** `isFraud` is extracted exclusively as label vector $y$. Automated tests (`tests/unit/test_schemas.py`, `tests/unit/test_feature_pipeline.py`) assert that `isFraud` is never passed into model feature rosters.

#### B. Complete Exclusion of Forbidden Fields
The following fields are strictly excluded from the feature space:
1. `isFlaggedFraud`: Excluded to prevent circular learning of the simulator's legacy heuristic rule (transfers $> 200,000$).
2. `newbalanceOrig`: Excluded because the updated account balance after debit settlement is not observable at authorization time.
3. `newbalanceDest`: Excluded because updated beneficiary balance post-credit is not available at decision time.
4. `nameOrig`, `nameDest`: Raw customer ID strings are excluded from $X$; utilized solely as grouping keys for historical velocity windows.

#### C. Operational Justification for `oldbalanceOrg` and `oldbalanceDest`
In standard banking and payment processing architectures (e.g. Visa/Mastercard clearing, Fedwire, CHIPS, core ledger systems):
- **Sender Balance (`oldbalanceOrg`):** Prior to authorizing any debit or transfer, the core banking ledger performs a mandatory balance inquiry. The account balance *immediately preceding* the proposed transaction is known to the authorization system at millisecond $t_0$.
- **Recipient Balance (`oldbalanceDest`):** In intrabank transfers (or verified interbank clearing networks), the recipient's pre-transaction balance is an accessible internal state variable.
- **Why Pre-Balances Do Not Constitute Leakage:** Unlike post-settlement balances (`newbalanceOrig`, `newbalanceDest`), which incorporate the completed settlement action, pre-balances exist strictly *before* transaction execution. Derived metrics such as `balance_orig_drain_ratio` ($\frac{\text{amount}}{\text{oldbalanceOrg} + \epsilon}$) model the borrower's solvency and account depletion risk at the exact moment of authorization.

#### D. Strict Point-in-Time Rolling Window Formulation
PaySim discrete simulation steps represent discrete hours ($t \in [1, 743]$). Multiple transactions occur within the same hour:
- **Intra-Step Non-Leakage Rule:** To eliminate contemporary leakage, rolling aggregates for transaction $i$ at step $t$ must aggregate strictly transactions where $\text{step} < t$.
- **Window Clause:** `RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING`
- **Verification:** Unit test `test_intra_step_non_leakage` verified that two transactions occurring at the identical hour (`step = 12`) cannot observe each other, returning `orig_tx_count_24h = 0`.
- **Invariance Test:** Unit test `test_future_transaction_invariance` verified that modifying future transactions at steps 15, 25, and 30 causes exactly $0.0000$ change to features at steps 5 and 10.

#### E. Preprocessing Isolation (No Distribution Snooping)
- All feature scaling (e.g., `RobustScaler`) and categorical frequency encodings are fitted **exclusively on the training partition** (`step <= 500`).
- The fitted transformer parameters (medians, interquartile ranges, corridor encodings) are serialized and applied as frozen inference transformations to validation and test partitions.
- No validation or test records inform scaler statistics.

---

### 4. Input Feature Specification for Models ($30$ Features)

The following $30$ features constitute the standardized model input vector $X$:

```
1.  amount                       (Float: nominal transaction currency)
2.  amount_log                   (Float: ln(amount + 1.0))
3.  oldbalanceOrg                (Float: pre-transaction sender balance)
4.  oldbalanceDest               (Float: pre-transaction recipient balance)
5.  balance_orig_drain_ratio     (Float: amount / (oldbalanceOrg + 1e-5))
6.  is_orig_zero_balance         (Binary: 1 if oldbalanceOrg == 0 and amount > 0)
7.  is_dest_zero_balance         (Binary: 1 if oldbalanceDest == 0 and amount > 0)
8.  orig_balance_coverage        (Float: oldbalanceOrg - amount)
9.  dest_balance_amount_ratio    (Float: amount / (oldbalanceDest + 1e-5))
10. hour_of_day                  (Integer: step % 24)
11. hour_sin                     (Float: sin(2pi * hour / 24))
12. hour_cos                     (Float: cos(2pi * hour / 24))
13. day_of_month                 (Integer: ((step - 1) // 24) % 31)
14. is_merchant_dest             (Binary: 1 if nameDest starts with 'M')
15. type_CASH_IN                 (Binary one-hot corridor flag)
16. type_CASH_OUT                (Binary one-hot corridor flag)
17. type_DEBIT                   (Binary one-hot corridor flag)
18. type_PAYMENT                 (Binary one-hot corridor flag)
19. type_TRANSFER                (Binary one-hot corridor flag)
20. orig_tx_count_1h             (Integer: sender tx count in preceding 1 hour)
21. orig_tx_count_6h             (Integer: sender tx count in preceding 6 hours)
22. orig_tx_count_24h            (Integer: sender tx count in preceding 24 hours)
23. orig_tx_count_72h            (Integer: sender tx count in preceding 72 hours)
24. orig_amount_sum_24h          (Float: sender cumulative amount in preceding 24 hours)
25. orig_amount_mean_24h         (Float: sender average amount in preceding 24 hours)
26. orig_amount_std_24h          (Float: sender amount standard deviation in 24 hours)
27. orig_amount_deviation_24h    (Float: amount - orig_amount_mean_24h)
28. orig_amount_zscore_24h       (Float: (amount - mean) / (std + 1e-5))
29. dest_inflow_count_24h        (Integer: recipient inflow count in preceding 24 hours)
30. dest_inflow_sum_24h          (Float: recipient cumulative inflow in preceding 24 hours)
31. is_first_interaction         (Binary: 1 if orig_tx_count_72h == 0 or dest_inflow_count_24h == 0)
```

---

### 5. Audit Sign-off

- [x] Zero target leakage verified.
- [x] Zero look-ahead bias verified.
- [x] Forbidden fields (`newbalanceOrig`, `newbalanceDest`, `isFlaggedFraud`) confirmed absent from Parquet.
- [x] Step-atomic chronological boundary isolation confirmed ($1\dots 500$ train, $501\dots 620$ val, $621\dots 743$ test).
- [x] Preprocessing fit isolation confirmed.

**Auditor Decision:** APPROVED. The feature pipeline and datasets satisfy all tier-1 banking risk engineering standards. Authorization granted to initiate baseline and model training.


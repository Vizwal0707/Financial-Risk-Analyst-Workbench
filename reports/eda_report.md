# PaySim Synthetic Financial Fraud Benchmark: Exploratory Data Analysis & Risk Report

**Portfolio Scope:** $6,362,620$ Transactions across $743$ Simulation Hours (~31 Days)  
**Total Monetary Volume:** $1,144,392,944,759.77  
**Fraud Exposure:** $12,056,415,427.84 (1.054% of portfolio currency)  
**Target Class:** `isFraud` (1 = Fraud, 0 = Legitimate)

---

### Executive Risk Summary & Key Quantitative Findings

1. **Extreme Asymmetric Class Imbalance:**
   - Fraud accounts for exactly $8,213$ out of $6,362,620$ transactions (**$0.1291\%$**), establishing an empirical class imbalance ratio of **$1:773.7$**.
   - Standard accuracy metrics are useless (a trivial majority-class classifier achieves $99.87\%$ accuracy while missing $100\%$ of monetary loss). Models must be evaluated using **Precision-Recall AUC (PR-AUC)**, **Precision@Top-K**, and **Fraud Value Captured (FVC)**.

2. **Strict Typology Corridor Isolation:**
   - Fraud is exclusively confined to two transaction corridors:
     - `TRANSFER`: $4,097$ frauds out of $532,909$ transactions (**$0.7688\%$** fraud rate).
     - `CASH_OUT`: $4,116$ frauds out of $2,237,500$ transactions (**$0.1840\%$** fraud rate).
   - In total conformance with PaySim's underlying agent mechanics, `PAYMENT`, `CASH_IN`, and `DEBIT` contain **zero fraud instances**.
   - *Financial Typology:* Adversaries compromise an account, initiate a high-value `TRANSFER` to a mule account, and immediately convert currency to liquid physical cash via `CASH_OUT`.

3. **Account Liquidation & Solvency Draining:**
   - In legitimate transactions, the median sender balance drain ratio ($rac{\text{amount}}{\text{oldbalanceOrg} + \epsilon}$) is near zero ($< 0.05$).
   - In fraudulent transfers, the drain ratio peaks sharply at **$1.000$**—the fraudster routinely drains the victim's account to \$0.00 in a single transaction.

4. **Statistical Divergence in Transaction Amounts:**
   - Median Legitimate Amount: **$74,195.27** (Mean: $174,230.26$)
   - Median Fraudulent Amount: **$441,423.44** (Mean: $1,467,967.30$)
   - Mann-Whitney U test confirms highly significant difference: $U = 1.72e+08$, $p = 0.00e+00$.
   - Kolmogorov-Smirnov two-sample test: $D = 0.4450$, $p = 0.00e+00$.

5. **Diurnal Pattern vs. Invariant Adversarial Attack Velocity:**
   - Legitimate customer traffic displays strong diurnal cyclicality, plunging between 01:00 and 06:00 and peaking between 12:00 and 18:00.
   - Fraud velocity remains stubbornly constant throughout all 24 hours of the day (automated botnets and programmed agent scripts). Consequently, the **effective fraud rate spikes during early morning hours** (02:00–05:00), making hour-of-day a potent risk modifier.

6. **Extreme Monetary Loss Concentration:**
   - The top $20\%$ of fraud transactions account for **$74.0\%$** of the total $\$12.06	ext{B}$ fraud exposure. Prioritizing high-value alerts in operational queues captures the majority of dollar loss even with constrained analyst bandwidth.

---

### Key Visualizations

| Figure | Description | File Link |
| :--- | :--- | :--- |
| **Figure 1** | Class Imbalance & Monetary Exposure | [class_imbalance.png](file:///Users/VizwalRathod/Desktop/Financial Fraud/reports/figures/class_imbalance.png) |
| **Figure 2** | Corridor Breakdown & Fraud Isolation | [corridor_fraud_distribution.png](file:///Users/VizwalRathod/Desktop/Financial Fraud/reports/figures/corridor_fraud_distribution.png) |
| **Figure 3** | Amount Distributions & Density | [amount_distribution_log.png](file:///Users/VizwalRathod/Desktop/Financial Fraud/reports/figures/amount_distribution_log.png) |
| **Figure 4** | Liquidation & Balance Drain Dynamics | [balance_drain_dynamics.png](file:///Users/VizwalRathod/Desktop/Financial Fraud/reports/figures/balance_drain_dynamics.png) |
| **Figure 5** | Diurnal Cycle vs. Attack Rate | [diurnal_transaction_velocity.png](file:///Users/VizwalRathod/Desktop/Financial Fraud/reports/figures/diurnal_transaction_velocity.png) |
| **Figure 6** | 743-Hour Activity & Partitions | [temporal_activity_743h.png](file:///Users/VizwalRathod/Desktop/Financial Fraud/reports/figures/temporal_activity_743h.png) |
| **Figure 7** | Cumulative Fraud Value (Lorenz) | [fraud_value_concentration.png](file:///Users/VizwalRathod/Desktop/Financial Fraud/reports/figures/fraud_value_concentration.png) |
| **Figure 8** | Feature Correlation Heatmap | [correlation_heatmap.png](file:///Users/VizwalRathod/Desktop/Financial Fraud/reports/figures/correlation_heatmap.png) |

---

### Dataset Limitations & Modeling Safeguards

1. **Synthetic Agent Boundary:** PaySim is an agent-based synthetic simulation. Fraud typologies reflect simulated script rules rather than human criminal ingenuity.
2. **Horizon Limitation:** 743 steps corresponds to 31 simulated calendar days. Longer-term quarterly or annual seasonality cannot be captured.
3. **Absence of Ground-Truth SAR / Chargeback Latency:** In real bank operations, fraud labels arrive days or weeks after transaction settlement (via customer disputes or chargebacks). PaySim labels transactions immediately, requiring historical evaluation to strictly respect temporal split boundaries.

*Report compiled by Data & AI Risk Architecture Pipeline.*

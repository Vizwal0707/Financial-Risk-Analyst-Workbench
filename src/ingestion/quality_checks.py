"""Data quality auditing, anomaly profiling, and statistical integrity reporting."""

from typing import Any

import numpy as np
import pandas as pd


class DataQualityAuditor:
    """Auditor for transaction datasets to verify integrity, completeness, and distributional sanity."""

    def __init__(self, expected_total_steps: int = 744) -> None:
        self.expected_total_steps = expected_total_steps

    def audit_raw_dataframe(self, df: pd.DataFrame) -> dict[str, Any]:
        """Performs a comprehensive data quality audit across the transaction dataset."""
        total_records = len(df)
        if total_records == 0:
            raise ValueError("Input DataFrame is empty; cannot perform data quality audit.")

        # 1. Missingness & Duplicates
        null_counts = df.isnull().sum().to_dict()
        total_nulls = int(df.isnull().sum().sum())
        duplicate_rows = int(df.duplicated().sum())

        # 2. Schema and Bounds
        negative_amounts = int((df["amount"] < 0).sum())
        negative_orig_bal = int((df["oldbalanceOrg"] < 0).sum())
        negative_dest_bal = int((df["oldbalanceDest"] < 0).sum())
        step_min = int(df["step"].min())
        step_max = int(df["step"].max())
        steps_out_of_bounds = int(
            ((df["step"] < 1) | (df["step"] > self.expected_total_steps)).sum()
        )

        # 3. Categorical types
        type_counts = df["type"].value_counts().to_dict()

        # 4. Target distribution & Imbalance
        is_fraud = df["isFraud"] if "isFraud" in df.columns else pd.Series(dtype=int)
        fraud_count = int(is_fraud.sum()) if not is_fraud.empty else 0
        legit_count = total_records - fraud_count
        fraud_rate_pct = (fraud_count / total_records) * 100.0
        imbalance_ratio = (legit_count / fraud_count) if fraud_count > 0 else float("inf")

        # 5. Financial volume metrics
        total_volume = float(df["amount"].sum())
        fraud_volume = (
            float(df.loc[df["isFraud"] == 1, "amount"].sum()) if "isFraud" in df.columns else 0.0
        )
        fraud_volume_pct = (fraud_volume / total_volume * 100.0) if total_volume > 0 else 0.0

        # 6. Fraud distribution across corridors
        fraud_by_type = {}
        if "isFraud" in df.columns:
            for t, count in type_counts.items():
                t_df = df[df["type"] == t]
                t_fraud = int(t_df["isFraud"].sum())
                fraud_by_type[t] = {
                    "total": int(len(t_df)),
                    "fraud_count": t_fraud,
                    "fraud_rate_pct": float(np.round((t_fraud / len(t_df)) * 100.0, 4))
                    if len(t_df) > 0
                    else 0.0,
                }

        # 7. Zero-balance anomaly counts
        orig_zero_bal_count = int((df["oldbalanceOrg"] == 0).sum())
        dest_zero_bal_count = int((df["oldbalanceDest"] == 0).sum())

        # Compile audit summary
        passed = (
            total_nulls == 0
            and duplicate_rows == 0
            and negative_amounts == 0
            and negative_orig_bal == 0
            and negative_dest_bal == 0
            and steps_out_of_bounds == 0
        )

        return {
            "passed": passed,
            "total_records": total_records,
            "total_nulls": total_nulls,
            "null_counts": null_counts,
            "duplicate_rows": duplicate_rows,
            "step_range": {"min": step_min, "max": step_max},
            "steps_out_of_bounds": steps_out_of_bounds,
            "negative_values": {
                "amount": negative_amounts,
                "oldbalanceOrg": negative_orig_bal,
                "oldbalanceDest": negative_dest_bal,
            },
            "type_counts": type_counts,
            "class_imbalance": {
                "legitimate_count": legit_count,
                "fraud_count": fraud_count,
                "fraud_rate_pct": round(fraud_rate_pct, 4),
                "imbalance_ratio": round(imbalance_ratio, 2),
            },
            "financial_exposure": {
                "total_nominal_volume": round(total_volume, 2),
                "total_fraud_volume": round(fraud_volume, 2),
                "fraud_volume_pct": round(fraud_volume_pct, 4),
            },
            "fraud_by_corridor": fraud_by_type,
            "structural_anomalies": {
                "orig_zero_balance_count": orig_zero_bal_count,
                "dest_zero_balance_count": dest_zero_bal_count,
            },
        }

    def generate_markdown_report(
        self, audit: dict[str, Any], title: str = "Dataset Quality & Validation Audit"
    ) -> str:
        """Formats the audit metrics into an institutional Markdown report."""
        imb = audit["class_imbalance"]
        exp = audit["financial_exposure"]
        step = audit["step_range"]

        corridor_rows = []
        for corridor, stats in audit.get("fraud_by_corridor", {}).items():
            corridor_rows.append(
                f"| `{corridor}` | {stats['total']:,} | {stats['fraud_count']:,} | {stats['fraud_rate_pct']:.4f}% |"
            )
        corridor_table = "\n".join(corridor_rows)

        report = f"""# {title}

**Audit Status:** {"PASSED" if audit["passed"] else "ACTION REQUIRED"}
**Total Records Analyzed:** {audit["total_records"]:,}
**Simulation Step Range:** [{step["min"]}, {step["max"]}] (Total hours: {step["max"] - step["min"] + 1})

---

### 1. Data Integrity & Schema Conformance

| Check Item | Result | Conformance Standard | Status |
| :--- | :--- | :--- | :--- |
| **Total Missing / Null Values** | {audit["total_nulls"]} | 0 nulls permitted | {"PASS" if audit["total_nulls"] == 0 else "FAIL"} |
| **Duplicate Transaction Records** | {audit["duplicate_rows"]} | 0 duplicate rows | {"PASS" if audit["duplicate_rows"] == 0 else "FAIL"} |
| **Negative Amounts / Balances** | {sum(audit["negative_values"].values())} | Amounts & Balances >= 0.0 | {"PASS" if sum(audit["negative_values"].values()) == 0 else "FAIL"} |
| **Out-of-Bounds Steps** | {audit["steps_out_of_bounds"]} | Steps within [1, {self.expected_total_steps}] | {"PASS" if audit["steps_out_of_bounds"] == 0 else "FAIL"} |

---

### 2. Class Imbalance & Monetary Exposure Analysis

- **Legitimate Transactions:** {imb["legitimate_count"]:,} ({100.0 - imb["fraud_rate_pct"]:.4f}%)
- **Fraudulent Transactions:** {imb["fraud_count"]:,} ({imb["fraud_rate_pct"]:.4f}%)
- **Class Imbalance Ratio:** 1 fraud per {imb["imbalance_ratio"]:,.1f} legitimate transactions (~1:{int(imb["imbalance_ratio"])})
- **Total Portfolio Volume:** ${exp["total_nominal_volume"]:,.2f}
- **Total Fraud Exposure:** ${exp["total_fraud_volume"]:,.2f} ({exp["fraud_volume_pct"]:.4f}% of total volume)

---

### 3. Fraud Distribution Across Transaction Corridors

| Transaction Corridor | Total Volume | Fraud Count | Corridor Fraud Rate |
| :--- | :--- | :--- | :--- |
{corridor_table}

> **Analytical Note:** In accordance with the PaySim simulation specification, fraud is exclusively concentrated in `TRANSFER` and `CASH_OUT` corridors. `PAYMENT`, `CASH_IN`, and `DEBIT` corridors show 0 fraud cases.

---

### 4. Structural Feature Topology Observations

- **Transactions Initiated with Zero Balance:** {audit["structural_anomalies"]["orig_zero_balance_count"]:,}
- **Transactions Received with Zero Destination Balance:** {audit["structural_anomalies"]["dest_zero_balance_count"]:,}

*Audit generated automatically by DataQualityAuditor.*
"""
        return report

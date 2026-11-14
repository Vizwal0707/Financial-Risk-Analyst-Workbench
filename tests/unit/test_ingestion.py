"""Unit tests for DuckDB analytical manager and DataQualityAuditor."""

from pathlib import Path

import pandas as pd

from src.ingestion.duckdb_manager import DuckDBManager
from src.ingestion.quality_checks import DataQualityAuditor


class TestDuckDBManager:
    """Tests DuckDB analytical engine connection, query execution, and profiling."""

    def test_in_memory_connection_and_query(self) -> None:
        """Asserts DuckDB executes SQL queries and returns pandas DataFrames."""
        db = DuckDBManager(db_path=":memory:")
        res = db.query("SELECT 42 AS answer, 'PaySim' AS benchmark")
        assert len(res) == 1
        assert res.iloc[0]["answer"] == 42
        assert res.iloc[0]["benchmark"] == "PaySim"
        db.close()

    def test_query_profiling_latency(self) -> None:
        """Asserts DuckDB query latency profiling produces valid metrics."""
        db = DuckDBManager(db_path=":memory:")
        db.execute("CREATE TABLE test_tx (id INT, amount DOUBLE)")
        db.execute("INSERT INTO test_tx VALUES (1, 100.0), (2, 250.0), (3, 500.0)")

        profile = db.profile_query_latency("SELECT AVG(amount) FROM test_tx", repetitions=5)
        assert "min_ms" in profile
        assert "mean_ms" in profile
        assert "p95_ms" in profile
        assert profile["min_ms"] >= 0.0
        assert profile["repetitions"] == 5
        db.close()

    def test_register_parquet_view(self, tmp_path: Path, sample_valid_raw_df: pd.DataFrame) -> None:
        """Asserts registering a Parquet file creates an accessible SQL view."""
        temp_parquet = tmp_path / "temp_tx.parquet"
        sample_valid_raw_df.to_parquet(temp_parquet)

        db = DuckDBManager(db_path=":memory:")
        db.register_parquet_view("v_transactions", str(temp_parquet))

        res = db.query("SELECT COUNT(*) AS total, SUM(isFraud) AS fraud_count FROM v_transactions")
        assert res.iloc[0]["total"] == 10
        assert res.iloc[0]["fraud_count"] == 2
        db.close()


class TestDataQualityAuditor:
    """Tests statistical data quality auditing, anomaly profiling, and reporting."""

    def test_audit_valid_dataset_passes(self, sample_valid_raw_df: pd.DataFrame) -> None:
        """Asserts compliant dataset passes all data quality checks."""
        auditor = DataQualityAuditor(expected_total_steps=744)
        audit = auditor.audit_raw_dataframe(sample_valid_raw_df)

        assert audit["passed"] is True
        assert audit["total_records"] == 10
        assert audit["total_nulls"] == 0
        assert audit["duplicate_rows"] == 0
        assert audit["class_imbalance"]["fraud_count"] == 2
        assert audit["class_imbalance"]["legitimate_count"] == 8
        assert audit["class_imbalance"]["fraud_rate_pct"] == 20.0
        assert audit["class_imbalance"]["imbalance_ratio"] == 4.0

    def test_audit_captures_nulls_and_fails(self, sample_valid_raw_df: pd.DataFrame) -> None:
        """Asserts auditor catches null values and marks passed=False."""
        df_nulls = sample_valid_raw_df.copy()
        df_nulls.loc[0, "amount"] = None

        auditor = DataQualityAuditor(expected_total_steps=744)
        audit = auditor.audit_raw_dataframe(df_nulls)
        assert audit["passed"] is False
        assert audit["total_nulls"] == 1

    def test_audit_captures_duplicates(self, sample_valid_raw_df: pd.DataFrame) -> None:
        """Asserts auditor catches duplicate rows."""
        df_dup = pd.concat([sample_valid_raw_df, sample_valid_raw_df.iloc[[0]]], ignore_index=True)
        auditor = DataQualityAuditor(expected_total_steps=744)
        audit = auditor.audit_raw_dataframe(df_dup)
        assert audit["passed"] is False
        assert audit["duplicate_rows"] == 1

    def test_markdown_report_generation(self, sample_valid_raw_df: pd.DataFrame) -> None:
        """Asserts auditor formats a readable markdown report."""
        auditor = DataQualityAuditor(expected_total_steps=744)
        audit = auditor.audit_raw_dataframe(sample_valid_raw_df)
        md = auditor.generate_markdown_report(audit, title="Test Audit Report")

        assert "# Test Audit Report" in md
        assert "PASSED" in md
        assert "Legitimate Transactions" in md
        assert "TRANSFER" in md

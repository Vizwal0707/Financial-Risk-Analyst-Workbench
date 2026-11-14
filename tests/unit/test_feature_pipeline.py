"""Unit tests for point-in-time feature engineering, invariance, parity, and partition atomicity."""

import numpy as np
import pandas as pd

from src.features.balance_transforms import BalanceTransformer
from src.features.duckdb_feature_pipeline import DuckDBFeaturePipeline
from src.features.pipeline import FinancialFeaturePipeline
from src.features.pyspark_feature_pipeline import PySparkFeaturePipeline
from src.features.transaction_transforms import TransactionTransformer


class TestPointInTimeInvariance:
    """Verifies that future transactions cannot alter past feature values."""

    def test_future_transaction_invariance(self, tmp_path) -> None:
        """Asserts modifying or adding future transactions does not change past features."""
        base_df = pd.DataFrame(
            {
                "step": [5, 10, 15],
                "type": ["TRANSFER", "CASH_OUT", "TRANSFER"],
                "amount": [1000.0, 2500.0, 3000.0],
                "nameOrig": ["C1001", "C1001", "C1001"],
                "oldbalanceOrg": [10000.0, 9000.0, 6500.0],
                "nameDest": ["M2001", "C3001", "C3001"],
                "oldbalanceDest": [0.0, 5000.0, 7500.0],
                "isFraud": [0, 0, 0],
            }
        )

        file_original = tmp_path / "stream_v1.parquet"
        base_df.to_parquet(file_original)

        pipeline = DuckDBFeaturePipeline()
        feats_v1 = pipeline.transform_df(str(file_original))

        # Capture step 5 and step 10 features
        row_step5_v1 = feats_v1[feats_v1["step"] == 5].iloc[0]
        row_step10_v1 = feats_v1[feats_v1["step"] == 10].iloc[0]

        # Scenario A: Modify future transaction at step 15 (change amount and destination)
        perturbed_df = base_df.copy()
        perturbed_df.loc[2, "amount"] = 999999.0
        perturbed_df.loc[2, "nameDest"] = "M9999"

        # Scenario B: Add future transactions at steps 25 and 30
        future_additions = pd.DataFrame(
            {
                "step": [25, 30],
                "type": ["TRANSFER", "CASH_OUT"],
                "amount": [50000.0, 80000.0],
                "nameOrig": ["C1001", "C1001"],
                "oldbalanceOrg": [3500.0, 0.0],
                "nameDest": ["C8888", "C9999"],
                "oldbalanceDest": [0.0, 1000.0],
                "isFraud": [1, 1],
            }
        )
        extended_df = pd.concat([perturbed_df, future_additions], ignore_index=True)

        file_extended = tmp_path / "stream_v2.parquet"
        extended_df.to_parquet(file_extended)

        feats_v2 = pipeline.transform_df(str(file_extended))
        row_step5_v2 = feats_v2[feats_v2["step"] == 5].iloc[0]
        row_step10_v2 = feats_v2[feats_v2["step"] == 10].iloc[0]

        # Assert 100% numerical invariance for historical records
        check_cols = [
            "orig_tx_count_1h",
            "orig_tx_count_6h",
            "orig_tx_count_24h",
            "orig_tx_count_72h",
            "orig_amount_sum_24h",
            "orig_amount_mean_24h",
            "orig_amount_std_24h",
            "balance_orig_drain_ratio",
            "orig_amount_deviation_24h",
        ]
        for col in check_cols:
            assert np.isclose(row_step5_v1[col], row_step5_v2[col], equal_nan=True), (
                f"Step 5 {col} changed!"
            )
            assert np.isclose(row_step10_v1[col], row_step10_v2[col], equal_nan=True), (
                f"Step 10 {col} changed!"
            )

    def test_intra_step_non_leakage(self, tmp_path) -> None:
        """Asserts transactions sharing the exact same step do not leak into each other's history."""
        # Two transactions by C1001 at the EXACT same step (step 12)
        intra_step_df = pd.DataFrame(
            {
                "step": [12, 12, 14],
                "type": ["TRANSFER", "CASH_OUT", "TRANSFER"],
                "amount": [500.0, 750.0, 1000.0],
                "nameOrig": ["C1001", "C1001", "C1001"],
                "oldbalanceOrg": [5000.0, 4500.0, 3750.0],
                "nameDest": ["C2001", "C2002", "C2001"],
                "oldbalanceDest": [100.0, 200.0, 600.0],
                "isFraud": [0, 0, 0],
            }
        )

        fpath = tmp_path / "intra_step.parquet"
        intra_step_df.to_parquet(fpath)

        pipeline = DuckDBFeaturePipeline()
        feats = pipeline.transform_df(str(fpath))

        # Both transactions at step 12 should see 0 prior transactions
        step12_rows = feats[feats["step"] == 12]
        for _, row in step12_rows.iterrows():
            assert row["orig_tx_count_24h"] == 0, (
                "Intra-step transaction leaked into current step history!"
            )
            assert row["orig_amount_sum_24h"] == 0.0

        # Step 14 transaction should see both step 12 transactions
        step14_row = feats[feats["step"] == 14].iloc[0]
        assert step14_row["orig_tx_count_24h"] == 2
        assert step14_row["orig_amount_sum_24h"] == 1250.0  # 500 + 750


class TestBackendParity:
    """Verifies that DuckDB SQL and PySpark window calculations produce identical output."""

    def test_duckdb_and_spark_emulation_parity(self, tmp_path) -> None:
        """Asserts DuckDB window aggregations match PySpark window specification exactly."""
        df = pd.DataFrame(
            {
                "step": [1, 2, 5, 10, 25, 26, 30],
                "type": [
                    "TRANSFER",
                    "CASH_OUT",
                    "TRANSFER",
                    "PAYMENT",
                    "CASH_OUT",
                    "TRANSFER",
                    "CASH_OUT",
                ],
                "amount": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
                "nameOrig": ["C1", "C1", "C1", "C2", "C1", "C1", "C2"],
                "oldbalanceOrg": [1000.0, 900.0, 700.0, 5000.0, 400.0, 1000.0, 4600.0],
                "nameDest": ["C3", "C3", "C4", "C3", "C3", "C4", "C3"],
                "oldbalanceDest": [50.0, 150.0, 0.0, 350.0, 750.0, 300.0, 1250.0],
                "isFraud": [0, 0, 0, 0, 0, 0, 0],
            }
        )

        fpath = tmp_path / "parity_test.parquet"
        df.to_parquet(fpath)

        # DuckDB Pipeline
        pipeline = DuckDBFeaturePipeline()
        duck_feats = pipeline.transform_df(str(fpath))

        # Spark Emulation
        spark_emul = PySparkFeaturePipeline.emulate_spark_window_parity(df)

        # Sort both by step, nameOrig, and amount to ensure identical row alignment
        duck_sorted = duck_feats.sort_values(by=["step", "nameOrig", "amount"]).reset_index(
            drop=True
        )
        # Add identifiers to spark_emul so it can be aligned identically
        spark_emul["step"] = df["step"]
        spark_emul["nameOrig"] = df["nameOrig"]
        spark_emul["amount"] = df["amount"]
        spark_sorted = spark_emul.sort_values(by=["step", "nameOrig", "amount"]).reset_index(
            drop=True
        )

        # Verify numerical parity across rolling window features
        metrics = [
            "orig_tx_count_1h",
            "orig_tx_count_6h",
            "orig_tx_count_24h",
            "orig_tx_count_72h",
            "orig_amount_sum_24h",
            "orig_amount_mean_24h",
            "orig_amount_std_24h",
            "dest_inflow_count_24h",
            "dest_inflow_sum_24h",
        ]
        for m in metrics:
            diff = np.abs(duck_sorted[m].values - spark_sorted[m].values)
            assert np.all(diff < 1e-4), f"Parity mismatch in {m}: max diff {np.max(diff)}"


class TestChronologicalPartitioning:
    """Tests atomic step-level partitioning (Train: 1-500, Val: 501-620, Test: 621-743)."""

    def test_partition_boundaries_and_atomicity(self, tmp_path) -> None:
        """Asserts each step belongs entirely to one partition with no partition boundary straddling."""
        df = pd.DataFrame(
            {
                "step": [1, 250, 500, 500, 501, 600, 620, 621, 700, 743],
                "type": ["TRANSFER"] * 10,
                "amount": [100.0] * 10,
                "nameOrig": [f"C{i}" for i in range(10)],
                "oldbalanceOrg": [1000.0] * 10,
                "nameDest": [f"D{i}" for i in range(10)],
                "oldbalanceDest": [500.0] * 10,
                "isFraud": [0] * 10,
            }
        )
        fpath = tmp_path / "part_test.parquet"
        df.to_parquet(fpath)

        pipeline = DuckDBFeaturePipeline()
        feats = pipeline.transform_df(str(fpath))

        train_steps = set(feats[feats["partition"] == "train"]["step"])
        val_steps = set(feats[feats["partition"] == "val"]["step"])
        test_steps = set(feats[feats["partition"] == "test"]["step"])

        assert max(train_steps) <= 500
        assert min(val_steps) >= 501
        assert max(val_steps) <= 620
        assert min(test_steps) >= 621
        assert max(test_steps) <= 743

        # Confirm disjoint sets (no overlap)
        assert len(train_steps.intersection(val_steps)) == 0
        assert len(val_steps.intersection(test_steps)) == 0
        assert len(train_steps.intersection(test_steps)) == 0

    def test_process_and_save_partitions(self, tmp_path) -> None:
        """Asserts process_and_save creates featurized dataset and discrete partition files."""
        df = pd.DataFrame(
            {
                "step": [10, 505, 650],
                "type": ["TRANSFER", "CASH_OUT", "TRANSFER"],
                "amount": [100.0, 200.0, 300.0],
                "nameOrig": ["C1", "C2", "C3"],
                "oldbalanceOrg": [1000.0, 2000.0, 3000.0],
                "nameDest": ["D1", "D2", "D3"],
                "oldbalanceDest": [100.0, 200.0, 300.0],
                "isFraud": [0, 1, 0],
            }
        )
        src_file = tmp_path / "raw.parquet"
        df.to_parquet(src_file)

        out_dir = tmp_path / "processed"
        pipeline = DuckDBFeaturePipeline()
        metrics = pipeline.process_and_save(str(src_file), output_dir=str(out_dir), save_partitions=True)

        assert metrics["total_records"] == 3
        assert (out_dir / "paysim_featurized.parquet").exists()
        assert (out_dir / "train.parquet").exists()
        assert (out_dir / "val.parquet").exists()
        assert (out_dir / "test.parquet").exists()


class TestFinancialFeaturePipeline:
    """Tests end-to-end scaler fitting and inference feature transformations."""

    def test_scaler_fitting_and_scaling(self, sample_valid_raw_df: pd.DataFrame) -> None:
        """Asserts RobustScaler fits strictly on training data and transforms clean feature matrix."""
        tx_trans = TransactionTransformer()
        bal_trans = BalanceTransformer()

        tx_f = tx_trans.transform(sample_valid_raw_df)
        bal_f = bal_trans.transform(sample_valid_raw_df)
        combined = pd.concat([tx_f, bal_f], axis=1)

        # Mock rolling metrics
        for col in [
            "orig_tx_count_1h",
            "orig_tx_count_6h",
            "orig_tx_count_24h",
            "orig_tx_count_72h",
            "orig_amount_sum_24h",
            "orig_amount_mean_24h",
            "orig_amount_std_24h",
            "orig_amount_deviation_24h",
            "orig_amount_zscore_24h",
            "dest_inflow_count_24h",
            "dest_inflow_sum_24h",
            "is_first_interaction",
        ]:
            combined[col] = 0.0

        pipeline = FinancialFeaturePipeline()
        pipeline.fit_scalers(combined)
        assert pipeline.is_fitted is True

        X_scaled = pipeline.scale_features(combined)
        assert X_scaled.shape == (10, len(pipeline.feature_columns))
        assert not np.isnan(X_scaled).any()

    def test_single_inference_matches_schema(self, sample_valid_transaction_dict: dict) -> None:
        """Asserts real-time single transaction scoring creates valid feature vector."""
        pipeline = FinancialFeaturePipeline()
        historical_cache = {
            "orig_tx_count_1h": 1,
            "orig_tx_count_24h": 3,
            "orig_amount_mean_24h": 150000.0,
            "orig_amount_std_24h": 25000.0,
            "dest_inflow_count_24h": 0,
        }

        feat_df = pipeline.transform_single_inference(
            transaction_dict=sample_valid_transaction_dict,
            historical_stats=historical_cache,
        )

        assert len(feat_df) == 1
        assert list(feat_df.columns) == pipeline.feature_columns
        assert feat_df["amount"].iloc[0] == 185000.50
        assert feat_df["orig_tx_count_24h"].iloc[0] == 3
        assert feat_df["is_first_interaction"].iloc[0] == 1  # Dest inflow is 0

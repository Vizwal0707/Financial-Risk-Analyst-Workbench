"""Unit tests for TransactionDataLoader and Parquet conversion."""

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.ingestion.loader import TransactionDataLoader
from src.ingestion.schemas import RawTransactionSchema


class TestTransactionDataLoader:
    """Tests chunked ingestion, serialization, and sampling utilities."""

    def test_convert_csv_to_parquet_and_read(
        self, tmp_path: Path, sample_valid_raw_df: pd.DataFrame
    ) -> None:
        """Asserts chunked CSV-to-Parquet conversion faithfully preserves all rows and schema."""
        csv_file = tmp_path / "test_raw.csv"
        parquet_file = tmp_path / "test_raw.parquet"
        sample_valid_raw_df.to_csv(csv_file, index=False)

        loader = TransactionDataLoader(raw_schema=RawTransactionSchema)
        total_rows = loader.convert_csv_to_parquet(
            csv_path=str(csv_file),
            parquet_output_path=str(parquet_file),
            chunk_size=5,
        )

        assert total_rows == 10
        assert parquet_file.exists()

        # Read back table and assert row count
        read_df = pq.read_table(str(parquet_file)).to_pandas()
        assert len(read_df) == 10
        assert list(read_df.columns) == list(sample_valid_raw_df.columns)

    def test_create_deterministic_ci_sample(
        self, tmp_path: Path, sample_valid_raw_df: pd.DataFrame
    ) -> None:
        """Asserts deterministic sampling generates valid stratified subset."""
        # Expand sample_valid_raw_df to 100 rows for sampling test
        df_large = pd.concat([sample_valid_raw_df] * 10, ignore_index=True)
        out_parquet = tmp_path / "ci_sample.parquet"

        loader = TransactionDataLoader(raw_schema=RawTransactionSchema)
        sample = loader.create_deterministic_ci_sample(
            source_df=df_large,
            output_parquet_path=str(out_parquet),
            sample_size=30,
            random_seed=42,
        )

        assert len(sample) == 30
        assert out_parquet.exists()
        assert "isFraud" in sample.columns
        assert sample["isFraud"].sum() > 0  # Guarantees positive fraud presence

"""Memory-efficient chunked data loader, Parquet conversion, and deterministic sampling."""

import logging
from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.schemas import RawTransactionSchema

logger = logging.getLogger(__name__)


class TransactionDataLoader:
    """Handles memory-efficient ingestion, schema validation, and Parquet serialization."""

    def __init__(self, raw_schema: type | None = RawTransactionSchema) -> None:
        self.raw_schema = raw_schema

    def stream_csv_chunks(
        self,
        csv_path: str,
        chunk_size: int = 500_000,
        validate_chunks: bool = True,
    ) -> Generator[pd.DataFrame, None, None]:
        """Yields validated DataFrame chunks from a large CSV file to avoid memory exhaustion."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Source CSV file not found at: {csv_path}")

        logger.info(f"Streaming CSV chunks (size={chunk_size:,}) from: {csv_path}")
        chunk_idx = 0
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            if validate_chunks and self.raw_schema is not None:
                chunk = self.raw_schema.validate(chunk)
            chunk_idx += 1
            logger.debug(f"Streamed and validated chunk {chunk_idx} ({len(chunk):,} rows)")
            yield chunk

    def convert_csv_to_parquet(
        self,
        csv_path: str,
        parquet_output_path: str,
        chunk_size: int = 500_000,
        compression: str = "snappy",
    ) -> int:
        """Converts a massive transaction CSV to Parquet via streaming chunked PyArrow writer."""
        out_path = Path(parquet_output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        total_rows = 0
        writer: pq.ParquetWriter | None = None

        try:
            for chunk in self.stream_csv_chunks(
                csv_path, chunk_size=chunk_size, validate_chunks=True
            ):
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(out_path), table.schema, compression=compression)
                writer.write_table(table)
                total_rows += len(chunk)
                logger.info(f"Progress: {total_rows:,} total rows serialized to Parquet")
        finally:
            if writer is not None:
                writer.close()

        logger.info(
            f"Completed CSV to Parquet conversion: {total_rows:,} rows written to {parquet_output_path}"
        )
        return total_rows

    def create_deterministic_ci_sample(
        self,
        source_df: pd.DataFrame,
        output_parquet_path: str,
        sample_size: int = 10_000,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """Extracts a stratified deterministic sample for CI tests and local lightweight prototyping.

        Preserves all fraud records up to target proportion or stratified distribution across corridors.
        """
        out_path = Path(output_parquet_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if "isFraud" not in source_df.columns:
            raise KeyError(
                "source_df must contain 'isFraud' target column for stratified sampling."
            )

        fraud_df = source_df[source_df["isFraud"] == 1]
        legit_df = source_df[source_df["isFraud"] == 0]

        total_fraud = len(fraud_df)
        total_source = len(source_df)
        fraud_ratio = total_fraud / total_source

        # Target sample allocation maintaining empirical fraud ratio with a guaranteed minimum
        target_fraud_count = max(int(sample_size * fraud_ratio), 15)
        target_fraud_count = min(target_fraud_count, total_fraud)
        target_legit_count = sample_size - target_fraud_count

        sampled_fraud = fraud_df.sample(n=target_fraud_count, random_state=random_seed)
        sampled_legit = legit_df.sample(n=target_legit_count, random_state=random_seed)

        sample = pd.concat([sampled_fraud, sampled_legit], ignore_index=True)
        # Sort chronologically by step to preserve temporal realism
        sample = sample.sort_values(by=["step"]).reset_index(drop=True)

        # Validate schema before persisting
        if self.raw_schema is not None:
            sample = self.raw_schema.validate(sample)

        table = pa.Table.from_pandas(sample, preserve_index=False)
        pq.write_table(table, str(out_path), compression="snappy")
        logger.info(
            f"Created deterministic CI sample fixture ({len(sample):,} rows, "
            f"{target_fraud_count} fraud) saved to {output_parquet_path}"
        )
        return sample

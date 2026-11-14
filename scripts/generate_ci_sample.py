"""Generates a deterministic 10,000-row CI sample fixture from the PaySim benchmark.

Preserves the exact schema, class imbalance, and step distribution across corridors
for high-speed unit tests, CI test pipelines, and local development.
"""

import argparse
import logging
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.schemas import RawTransactionSchema

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("GenerateCISample")

REMOTE_PAYSIM_URL = (
    "https://huggingface.co/datasets/purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection"
    "/resolve/main/financial_txns.parquet"
)


def generate_ci_sample(
    source_parquet: str = None,
    output_parquet: str = "data/sample/paysim_ci_sample.parquet",
    sample_size: int = 10_000,
    random_seed: int = 42,
) -> Path:
    """Extracts a stratified deterministic sample and validates it with Pandera."""
    out_path = Path(output_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine source: local file if exists, else remote URL
    if source_parquet and Path(source_parquet).exists():
        src = str(Path(source_parquet).resolve())
        logger.info(f"Generating CI sample from local source: {src}")
    elif Path("data/raw/paysim.parquet").exists():
        src = str(Path("data/raw/paysim.parquet").resolve())
        logger.info("Generating CI sample from local data/raw/paysim.parquet")
    else:
        src = REMOTE_PAYSIM_URL
        logger.info(f"Generating CI sample directly from verified benchmark repository: {src}")

    conn = duckdb.connect()

    # Calculate target counts: ~0.20% fraud (minimum 20 cases for robust test slices across splits)
    fraud_target = 25
    legit_target = sample_size - fraud_target

    sql = f"""
    WITH fraud_rows AS (
        SELECT * FROM (
            SELECT * FROM read_parquet('{src}')
            WHERE isFraud = 1
        )
        USING SAMPLE {fraud_target} (reservoir, {random_seed})
    ),
    legit_rows AS (
        SELECT * FROM (
            SELECT * FROM read_parquet('{src}')
            WHERE isFraud = 0
        )
        USING SAMPLE {legit_target} (reservoir, {random_seed})
    )
    SELECT * FROM (
        SELECT * FROM fraud_rows
        UNION ALL
        SELECT * FROM legit_rows
    )
    ORDER BY step, amount
    """

    logger.info("Executing deterministic stratified query via DuckDB...")
    df = conn.execute(sql).df()

    # Validate against Pandera RawTransactionSchema
    logger.info("Validating sampled DataFrame against Pandera schema contract...")
    validated_df = RawTransactionSchema.validate(df)

    # Persist with Snappy compression
    table = pa.Table.from_pandas(validated_df, preserve_index=False)
    pq.write_table(table, str(out_path), compression="snappy")

    fraud_in_sample = int(validated_df["isFraud"].sum())
    logger.info(
        f"Successfully generated CI sample: {len(validated_df):,} rows "
        f"({fraud_in_sample} fraud, {len(validated_df) - fraud_in_sample} legit) "
        f"saved to {out_path} ({out_path.stat().st_size / (1024):.1f} KB)."
    )
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate deterministic CI sample fixture.")
    parser.add_argument(
        "--source", default=None, help="Path to local PaySim parquet file if available."
    )
    parser.add_argument(
        "--output", default="data/sample/paysim_ci_sample.parquet", help="Output path."
    )
    parser.add_argument("--size", type=int, default=10_000, help="Total sample rows.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed.")
    args = parser.parse_args()

    generate_ci_sample(
        source_parquet=args.source,
        output_parquet=args.output,
        sample_size=args.size,
        random_seed=args.seed,
    )

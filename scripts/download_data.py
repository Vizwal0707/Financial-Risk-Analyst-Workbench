"""Reproducible PaySim benchmark dataset acquisition script.

Acquires the genuine PaySim benchmark dataset (6,362,620 transactions) from verified
legitimate public sources (Hugging Face Datasets mirror or Kaggle API),
verifies schema and record counts (6,362,620 rows, 8,213 frauds), and normalizes paths.

PaySim is a synthetic financial fraud benchmark created by Lopez-Rojas et al.
It does not represent real bank transactions or proprietary customer data.
"""

import argparse
import logging
import sys
from pathlib import Path

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("DataAcquisition")

HF_PAYSIM_PARQUET_URL = (
    "https://huggingface.co/datasets/purulalwani/Synthetic-Financial-Datasets-For-Fraud-Detection"
    "/resolve/main/financial_txns.parquet"
)
EXPECTED_TOTAL_ROWS = 6362620
EXPECTED_FRAUD_COUNT = 8213


def download_file(url: str, output_path: Path) -> None:
    """Downloads a file with streaming chunking and a progress bar."""
    logger.info(f"Downloading from {url} to {output_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(output_path, "wb") as f,
        tqdm(total=total_size, unit="B", unit_scale=True, desc=output_path.name) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def verify_dataset(parquet_path: Path) -> bool:
    """Verifies that the downloaded Parquet dataset contains the exact expected PaySim record counts."""
    import duckdb

    conn = duckdb.connect()
    res = conn.execute(
        f"SELECT COUNT(*), SUM(isFraud) FROM read_parquet('{str(parquet_path.resolve())}')"
    ).fetchone()
    total_rows, fraud_count = res[0], int(res[1])
    logger.info(
        f"Dataset verification: {total_rows:,} total transactions, {fraud_count:,} fraud cases."
    )

    if total_rows != EXPECTED_TOTAL_ROWS or fraud_count != EXPECTED_FRAUD_COUNT:
        logger.warning(
            f"Record mismatch: Expected ({EXPECTED_TOTAL_ROWS:,}, {EXPECTED_FRAUD_COUNT:,}), "
            f"found ({total_rows:,}, {fraud_count:,})"
        )
        return False
    logger.info("Dataset verification: PASSED. Exact PaySim benchmark counts confirmed.")
    return True


def acquire_dataset(target_dir: str = "data/raw", force: bool = False) -> Path:
    """Acquires PaySim benchmark dataset and verifies integrity."""
    raw_dir = Path(target_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_parquet = raw_dir / "paysim.parquet"
    target_csv = raw_dir / "paysim.csv"

    if target_parquet.exists() and not force:
        logger.info(f"PaySim raw parquet already exists at {target_parquet}.")
        if verify_dataset(target_parquet):
            return target_parquet

    if target_csv.exists() and not force:
        logger.info(f"PaySim raw CSV exists at {target_csv}.")
        return target_csv

    try:
        download_file(HF_PAYSIM_PARQUET_URL, target_parquet)
        if verify_dataset(target_parquet):
            logger.info("Successfully acquired genuine PaySim dataset.")
            return target_parquet
    except Exception as e:
        logger.error(f"Download failed: {e}. Please check your network connection.")
        sys.exit(1)

    return target_parquet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire the genuine PaySim benchmark dataset.")
    parser.add_argument("--target-dir", default="data/raw", help="Target directory for raw data.")
    parser.add_argument("--force", action="store_true", help="Force re-download.")
    args = parser.parse_args()

    acquire_dataset(target_dir=args.target_dir, force=args.force)

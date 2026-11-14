"""Execution script for point-in-time feature engineering pipeline.

Processes raw PaySim Parquet dataset, computes real-time, solvency, and rolling velocity features,
partitions data chronologically into Train (1-500), Validation (501-620), and Test (621-743),
and writes feature parquet artifacts.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from src.features.duckdb_feature_pipeline import DuckDBFeaturePipeline

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("BuildFeatures")


def run_feature_pipeline(
    source_parquet: str = "data/raw/paysim.parquet",
    output_dir: str = "data/processed",
    metrics_path: str = "reports/feature_engineering_metrics.json",
    save_partitions: bool = True,
) -> dict:
    """Executes the DuckDB feature engineering pipeline and records performance metrics."""
    logger.info(f"Starting feature engineering on: {source_parquet}")
    start_total = time.perf_counter()

    pipeline = DuckDBFeaturePipeline(
        train_max_step=500,
        val_max_step=620,
        test_max_step=743,
    )

    metrics = pipeline.process_and_save(
        source_parquet_path=source_parquet,
        output_dir=output_dir,
        save_partitions=save_partitions,
    )

    metrics["total_pipeline_time_seconds"] = round(time.perf_counter() - start_total, 2)

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved feature engineering metrics to: {metrics_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build point-in-time financial features.")
    parser.add_argument("--source", default="data/raw/paysim.parquet", help="Source raw parquet.")
    parser.add_argument("--output-dir", default="data/processed", help="Processed data directory.")
    parser.add_argument(
        "--metrics", default="reports/feature_engineering_metrics.json", help="Metrics JSON path."
    )
    parser.add_argument(
        "--no-partitions", action="store_true", help="Skip saving individual partition files."
    )
    args = parser.parse_args()

    run_feature_pipeline(
        source_parquet=args.source,
        output_dir=args.output_dir,
        metrics_path=args.metrics,
        save_partitions=not args.no_partitions,
    )

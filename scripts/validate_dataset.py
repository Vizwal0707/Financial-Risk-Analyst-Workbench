"""Full dataset validation, schema conformance auditing, and metadata report generation."""

import argparse
import json
import logging
from pathlib import Path

import duckdb

from src.ingestion.quality_checks import DataQualityAuditor
from src.ingestion.schemas import RawTransactionSchema

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("ValidateDataset")


def validate_dataset(
    data_path: str,
    output_report_path: str = "reports/dataset_validation_report.md",
    output_json_path: str = "reports/dataset_validation_metrics.json",
    limit: int = None,
) -> dict:
    """Performs full statistical and schema validation and produces an audit report."""
    path_obj = Path(data_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Target data file not found at: {data_path}")

    logger.info(f"Loading transaction dataset for validation from: {data_path}...")
    conn = duckdb.connect()

    if data_path.endswith(".parquet"):
        sql = f"SELECT * FROM read_parquet('{str(path_obj.resolve())}')"
    elif data_path.endswith(".csv"):
        sql = f"SELECT * FROM read_csv_auto('{str(path_obj.resolve())}')"
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    if limit is not None:
        sql += f" LIMIT {limit}"

    df = conn.execute(sql).df()
    logger.info(f"Loaded {len(df):,} transactions for data quality audit.")

    # 1. Pandera schema contract validation
    logger.info("Executing Pandera schema contract validation...")
    try:
        RawTransactionSchema.validate(df)
        logger.info("Pandera schema check: PASSED (all types, constraints, non-nullity verified).")
    except Exception as e:
        logger.error(f"Pandera schema check FAILED: {e}")
        raise

    # 2. Comprehensive statistical and distributional audit
    logger.info("Performing deep statistical and integrity audit...")
    auditor = DataQualityAuditor(expected_total_steps=744)
    audit_results = auditor.audit_raw_dataframe(df)

    # 3. Format and save outputs
    report_md = auditor.generate_markdown_report(
        audit_results,
        title=f"PaySim Transaction Dataset Quality Audit ({path_obj.name})",
    )

    Path(output_report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w") as f:
        f.write(report_md)
    logger.info(f"Saved Markdown validation report to: {output_report_path}")

    Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(audit_results, f, indent=2)
    logger.info(f"Saved JSON validation metrics to: {output_json_path}")

    return audit_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit and validate a financial transaction dataset."
    )
    parser.add_argument(
        "--data", default="data/sample/paysim_ci_sample.parquet", help="Path to Parquet or CSV."
    )
    parser.add_argument(
        "--report",
        default="reports/dataset_validation_report.md",
        help="Markdown report destination.",
    )
    parser.add_argument(
        "--metrics",
        default="reports/dataset_validation_metrics.json",
        help="JSON metrics destination.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    args = parser.parse_args()

    validate_dataset(
        data_path=args.data,
        output_report_path=args.report,
        output_json_path=args.metrics,
        limit=args.limit,
    )

"""Project build & integrity verification pipeline."""

import compileall
from pathlib import Path
import sys


def main() -> None:
    print("=" * 80)
    print("BUILDING: Financial Fraud Detection & Risk Intelligence Platform")
    print("=" * 80)

    root = Path(".")
    print("\n[1/4] Bytecode Compilation & Syntax Verification...")
    success = compileall.compile_dir("src", force=True, quiet=1)
    success &= compileall.compile_dir("dashboard", force=True, quiet=1)
    success &= compileall.compile_dir("configs", force=True, quiet=1)
    if not success:
        print("❌ Syntax error during bytecode compilation.")
        sys.exit(1)
    print("✓ All Python modules in src/, dashboard/, and configs/ compiled cleanly.")

    print("\n[2/4] Verifying Trained Model Artifacts...")
    model_files = [
        "models/xgboost_best.joblib",
        "models/calibrator_best.joblib",
        "models/autoencoder_best.pt",
        "models/autoencoder_scaler.joblib",
        "models/hybrid_scorer_best.joblib",
        "models/decision_policy_thresholds.json",
    ]
    for mf in model_files:
        p = root / mf
        if not p.exists():
            print(f"❌ Missing required model artifact: {mf}")
            sys.exit(1)
        print(f"  ✓ {mf} ({p.stat().st_size:,} bytes)")

    print("\n[3/4] Verifying Pre-scored Evaluation Datasets...")
    datasets = [
        "data/processed/val_evaluation_scored.parquet",
        "data/processed/test_evaluation_scored.parquet",
    ]
    for ds in datasets:
        p = root / ds
        if not p.exists():
            print(f"❌ Missing required evaluation dataset: {ds}")
            sys.exit(1)
        print(f"  ✓ {ds} ({p.stat().st_size:,} bytes)")

    print("\n[4/4] Verifying Production Deployment Assets...")
    deployment_assets = [
        "Dockerfile.api",
        "Dockerfile.streamlit",
        "docker-compose.yml",
        ".dockerignore",
        ".env.example",
        "DEPLOYMENT.md",
    ]
    for da in deployment_assets:
        p = root / da
        if not da in [x.name for x in root.iterdir()]:
            print(f"❌ Missing deployment asset: {da}")
            sys.exit(1)
        print(f"  ✓ {da}")

    print("\n" + "=" * 80)
    print("✓ BUILD COMPLETED SUCCESSFULLY")
    print("  Backend API:    uvicorn src.api.main:app --port 8000")
    print("  Streamlit UI:   streamlit run dashboard/app.py --server.port 8501")
    print("  Test Suite:     npm test  (or: pytest tests/)")
    print("  Docker Stack:   docker compose up -d")
    print("=" * 80)


if __name__ == "__main__":
    main()


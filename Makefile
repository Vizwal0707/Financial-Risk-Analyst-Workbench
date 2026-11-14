.PHONY: help install test lint format sample validate download eda features api dashboard benchmark docker-build docker-up clean

PYTHON = ./.venv/bin/python
PIP = ./.venv/bin/pip
PYTEST = ./.venv/bin/pytest

help:
	@echo "Financial Fraud Detection & Risk Intelligence Platform"
	@echo "Available targets:"
	@echo "  install      - Install pinned dependencies"
	@echo "  sample       - Generate deterministic 10,000-row CI sample"
	@echo "  download     - Acquire full PaySim benchmark dataset"
	@echo "  validate     - Run Pandera and data quality audit"
	@echo "  eda          - Generate EDA visualizations and research report"
	@echo "  features     - Build point-in-time features & partition splits"
	@echo "  api          - Start local FastAPI transaction scoring service"
	@echo "  dashboard    - Start Streamlit Institutional Risk Workbench"
	@echo "  benchmark    - Run SLA latency and batch throughput benchmark"
	@echo "  docker-build - Build multi-stage production Docker image"
	@echo "  docker-up    - Launch container stack with PostgreSQL database"
	@echo "  test         - Run pytest test suite with coverage"
	@echo "  lint         - Run code linter"
	@echo "  clean        - Remove temporary caches and artifacts"

install:
	$(PIP) install -r requirements.txt

sample:
	PYTHONPATH=. $(PYTHON) scripts/generate_ci_sample.py

download:
	PYTHONPATH=. $(PYTHON) scripts/download_data.py

validate:
	PYTHONPATH=. $(PYTHON) scripts/validate_dataset.py

eda:
	PYTHONPATH=. $(PYTHON) scripts/generate_eda_visualizations.py
	PYTHONPATH=. $(PYTHON) scripts/build_notebooks.py

features:
	PYTHONPATH=. $(PYTHON) scripts/build_features.py

api:
	MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. $(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2

dashboard:
	MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. $(PYTHON) -m streamlit run dashboard/app.py --server.port 8501 --server.address 127.0.0.1

benchmark:
	MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. $(PYTHON) scripts/benchmark_api.py --n-single 1000

docker-build:
	docker build -t financial-fraud-api:latest .

docker-up:
	docker compose up -d

test:
	MPLCONFIGDIR=/tmp/matplotlib PYTHONPATH=. $(PYTEST) tests/ -v --cov=src --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache



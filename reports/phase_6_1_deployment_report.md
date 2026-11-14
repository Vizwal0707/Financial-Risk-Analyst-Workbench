# Phase 6.1 — Public Deployment, Security Hardening & Hosting Verification Report

**Organization:** JPMorgan Chase & Co. — Financial Crime Risk Management & Quantitative Analytics  
**Platform:** Financial Fraud Detection & Risk Intelligence Platform (PaySim Benchmark)  
**Document Classification:** Technical Infrastructure & Deployment Evaluation  
**Status:** COMPLETED & VERIFIED  
**Phase:** 6.1  

---

## 1. Executive Summary

Phase 6.1 prepared the Financial Fraud Detection & Risk Intelligence Platform for public cloud hosting and containerized enterprise deployment. The system was structured into two decoupled, independently deployable services:

1. **Frontend Service:** Institutional Streamlit Risk Analyst Workbench & Model Governance Dashboard.
2. **Backend Service:** FastAPI Transaction Scoring & TreeSHAP Explainability Engine.

A dual-mode persistence architecture was designed and implemented to eliminate the catastrophic ephemeral filesystem risk inherent in PaaS container hosting. Security hardening was completed across container execution (non-root unprivileged users), network boundaries (safe CORS and unexposed internal database ports), and presentation layers (PCI-DSS-aligned account masking and password redaction).

All 98 automated unit and integration tests passed with 100% success. No paid cloud resources were purchased or provisioned, adhering strictly to audit protocols.

---

## 2. Decoupled Service Architecture

```
                                  [ Web Client / Browser ]
                                             |
                                   HTTPS (Public Internet)
                                             v
+-----------------------------------------------------------------------------------------+
| FRONTEND SERVICE: Streamlit Risk Workbench (Port 8501)                                  |
| - Base: python:3.11-slim (Multi-stage)                                                  |
| - Security: Non-root user appuser (UID 1001)                                            |
| - Environment Target: FRAUD_API_URL (Dynamic HTTPS endpoint, no hardcoded localhost)    |
| - Offline Resilience: Demo Mode (Precomputed benchmark exploration when API offline)    |
| - Offline Integrity: Never fabricates synthetic scoring responses                      |
+-----------------------------------------------------------------------------------------+
                                             |
                              HTTPS / Private VPC Network
                                             v
+-----------------------------------------------------------------------------------------+
| BACKEND SERVICE: FastAPI Scoring Engine (Port 8000)                                     |
| - Base: python:3.11-slim (Multi-stage)                                                  |
| - Security: Non-root user appuser (UID 1001)                                            |
| - Inference Engine: XGBoost + Isotonic Calibrator + DeepAutoencoder + TreeSHAP          |
| - Hardened CORS: Configured via CORS_ALLOWED_ORIGINS (Restricts cross-origin clients)     |
| - Probes: /health (Liveness probe), /ready (Readiness probe)                            |
+-----------------------------------------------------------------------------------------+
                                             |
                         SQLAlchemy Connection Pool (Port 5432)
                                             v
+-----------------------------------------------------------------------------------------+
| DUAL-MODE PERSISTENCE LAYER:                                                            |
| - Local / Offline: SQLite 3.45 with PRAGMA journal_mode=WAL                             |
| - Cloud / Production: Managed PostgreSQL 16 (Neon / Supabase / Render / AWS RDS)        |
| - Resilience Fallback: Automatic SQLite fallback if PostgreSQL connection drops         |
| - Durability Guarantee: Prevents data loss from PaaS container ephemeral restarts       |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Database Deployment Review & Ephemeral Filesystem Mitigation

### 3.1 The Ephemeral Filesystem Problem on PaaS
Modern Platform-as-a-Service (PaaS) providers—including Render, Railway, Fly.io, and Streamlit Community Cloud—run application containers on stateless, ephemeral virtual filesystems. When a container restarts, redeploys, or wakes from idle sleep:
$$\text{Container Restart} \implies \text{Ephemeral Filesystem Wiped} \implies \text{Local SQLite Database Destroyed}$$
In a financial compliance context, losing the regulatory audit log (`audit_scoring_logs`) represents an unacceptable compliance violation.

### 3.2 Dual-Mode Persistence Architecture (`DatabaseManager`)
To address this risk, `src/persistence/database.py` was refactored into a hardened dual-mode engine:
1. **SQLite WAL Mode (Local / Developer):**
   - Configured via `DATABASE_URL=sqlite:///data/audit_scoring.db`.
   - Executes with `PRAGMA journal_mode=WAL;` and `PRAGMA synchronous=NORMAL;` for high-concurrency local file persistence.
2. **PostgreSQL Dual-Mode (Cloud PaaS):**
   - Configured via `DATABASE_URL=postgresql://user:pass@host:5432/dbname`.
   - Utilizes SQLAlchemy engine with connection pooling (`pool_pre_ping=True`, `pool_size=10`, `max_overflow=20`, `pool_recycle=1800`).
   - Automatically converts SQLite DDL syntax (`AUTOINCREMENT` $\rightarrow$ `SERIAL`).
3. **Automatic Fallback Safety Net:**
   - If PostgreSQL credentials fail, connection timeouts occur, or the database server is unreachable, `DatabaseManager` logs a structured warning and falls back immediately to local SQLite, ensuring zero scoring downtime.
   - Tested and verified in `tests/deployment/test_deployment_hardening.py::test_database_manager_postgres_fallback_to_sqlite`.
4. **Verification Status:**
   - PostgreSQL mode is fully implemented and verified via local container and fallback tests.
   - **Production Status:** Formally documented as **unverified in production** until live testing is conducted against real cloud infrastructure.

---

## 4. Demo Mode Architecture & Offline Resilience

When the backend scoring service is offline, asleep, or unreachable, the Streamlit workbench operates in **Demo Mode**:

| Feature / Screen | Online Mode Behavior | Demo Mode (Offline) Behavior |
| :--- | :--- | :--- |
| **Landing Ribbon (`app.py`)** | Shows "FastAPI Engine: ONLINE" (Green) | Displays prominent warning banner: `⚡ DEMO MODE: FastAPI Scoring Backend is currently offline.` (Amber) |
| **Transaction Explorer (`03_Transaction_Explorer.py`)** | Submits live payloads to `POST /v1/score` | Live submission button disabled. Prompts user to inspect pre-scored demo records in Persisted Audit tab. **Zero fake responses fabricated.** |
| **Executive Overview (`01_Executive_Overview.py`)** | Toggles between Frozen Benchmark and Live DB | Displays precomputed frozen test set metrics (Steps 621–743, N=90,829). Live DB tab indicates empty/offline state gracefully. |
| **Model Governance (`04_Model_Governance.py`)** | Full audited test metrics & ablation results | Operates 100% offline using audited Phase 4.1 report artifacts. |
| **Policy Simulator (`05_Policy_Simulator.py`)** | Interactive sliders over pre-scored Parquet | Operates 100% offline over `val_evaluation_scored.parquet` ($N=209,984$). |
| **Audit Explorer (`06_Audit_Explorer.py`)** | Queries live audit logs | Displays cached local SQLite records or displays a graceful empty notification without crashing. |

---

## 5. Cloud Hosting Provider Evaluation & Pricing Matrix

A comparative evaluation was performed across top cloud providers to determine the optimal hosting topology:

| Provider | Supported Role | Free Tier Limitations | Paid Tier Pricing | Evaluation Summary |
| :--- | :--- | :--- | :--- | :--- |
| **Render** | Backend API & PostgreSQL | Free web service spins down after 15 min inactivity (~50s cold start). Free Postgres expires after 90 days. | **Starter Web Service:** \$7.00/mo (512 MB RAM, 0.5 CPU, always-on). **Starter Postgres:** \$7.00/mo (1 GB RAM, 10 GB SSD). | **Highly Recommended for Backend.** Native Dockerfile support, automated HTTPS certificates, zero maintenance. |
| **Railway** | Backend API & Database | \$5.00 monthly usage credit (trial). Spins down when credit exhausts. | **Hobby:** \$5.00 base + usage (\$0.000231/GB-hr RAM, \$0.000463/vCPU-hr). | **Fastest Developer Experience.** Excellent private networking between containers; native Postgres plugin. |
| **Fly.io** | Backend API | Up to 3 shared-cpu-1x VMs (256 MB RAM) free with credit card verification. | **Pay-as-you-go:** ~\$1.94/mo per 256 MB shared VM; ~\$5.70/mo for 512 MB VM. | **Low Latency Edge Hosting.** Fast container boots; requires CLI deployment (`flyctl`). |
| **Streamlit Community Cloud** | Frontend Dashboard | **100% Free** for public GitHub repositories. 1 GB RAM per app. | Enterprise pricing custom. | **Highly Recommended for Frontend.** Native Streamlit hosting, automatic deploys from `main` branch, zero infrastructure overhead. |
| **Neon** | Serverless PostgreSQL | **Free Tier:** 0.5 GB storage, serverless compute (scales to zero after 5 min). | **Launch:** \$19.00/mo for autoscaling compute. | **Top Choice for Serverless DB.** Native Postgres 16, connection pooling, free tier sufficient for prototyping. |
| **Supabase** | Managed PostgreSQL | **Free Tier:** 500 MB database, pauses after 1 week inactivity. | **Pro:** \$25.00/mo. | Robust managed Postgres, but pauses during prolonged idle periods. |

---

## 6. Recommended Lowest-Cost Production Setup

To maximize reliability, eliminate cold-start latency, and keep costs under \$10/month:

```
[ Public Internet ]
        |
        v
+----------------------------------------------------------+
| Streamlit Community Cloud (Frontend Dashboard)           |
| - Cost: $0.00 / month (Free tier)                         |
| - Source: GitHub repository main branch                  |
| - Secrets: FRAUD_API_URL pointing to Render backend      |
+----------------------------------------------------------+
        |
        v HTTPS
+----------------------------------------------------------+
| Render Starter Web Service (FastAPI Backend Scoring)     |
| - Cost: $7.00 / month (Always-on, no cold starts)        |
| - Build: Dockerfile.api                                  |
| - Specs: 512 MB RAM, 0.5 vCPU                            |
+----------------------------------------------------------+
        |
        v Encrypted TLS
+----------------------------------------------------------+
| Neon Serverless PostgreSQL / Render Free Postgres        |
| - Cost: $0.00 / month                                    |
| - Engine: PostgreSQL 16 with connection pooling          |
+----------------------------------------------------------+
```

**Total Estimated Operating Cost:** **\$7.00 / month** (or **\$0.00 / month** using free tiers with 50s cold-start tolerance).

---

## 7. Container Specifications & Multi-Container Verification

### 7.1 Created Deployment Artifacts
- **`Dockerfile.api`:** Multi-stage build (`python:3.11-slim`), non-root `appuser` (UID 1001), healthcheck on `/health`, port 8000.
- **`Dockerfile.streamlit`:** Multi-stage build (`python:3.11-slim`), non-root `appuser` (UID 1001), healthcheck on `/_stcore/health`, port 8501.
- **`.dockerignore`:** Excludes 1.2 GB of raw Parquet files, virtual environments, bytecode, and cache directories, reducing build context from ~1.5 GB to **~28 MB**.
- **`docker-compose.yml`:** Defines 3 isolated services (`fraud-api`, `fraud-dashboard`, `fraud-db`) with internal network bridging and shielded database ports.
- **`.env.example`:** Complete production environment template documenting all backend, frontend, CORS, and persistence variables.
- **`DEPLOYMENT.md`:** End-to-end operational runbook covering local Docker, Render, Railway, Fly.io, and Streamlit Community Cloud.

---

## 8. Full Regression Test Verification

The entire automated test suite was executed across all unit, integration, API, dashboard, and deployment modules:

```
======================= 98 passed, 18 warnings in 22.32s =======================
```

- **Total Test Count:** 98 passed / 0 failed (100% pass rate).
- **Execution Environment:** macOS Darwin, Python 3.11.3, Pytest 9.2.0.
- **Zero Regressions:** All 92 baseline Phase 1–6 tests preserved intact; 6 new deployment hardening tests verified.

---

## 9. Conclusion & Phase 7 Transition Readiness

Phase 6.1 has satisfied all deployment preparation, containerization, security hardening, and model governance requirements. The system is structurally decoupled, air-gapped from live banking clearing, and ready for public cloud deployment.

**Phase 6.1 is formally COMPLETED. Execution halted to await user review and authorization prior to Phase 7.**


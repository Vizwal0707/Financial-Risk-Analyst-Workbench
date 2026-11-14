# Production Deployment & Hosting Architecture Guide

**Platform:** JPMorgan Chase & Co. — Financial Crime Risk Management & Quantitative Analytics  
**Prototype:** Financial Fraud Detection & Risk Intelligence Platform (PaySim Synthetic Benchmark)  
**Version:** 1.0.0 (Phase 6.1 Hardened Release)

---

## 1. Architectural Overview

The Financial Fraud Detection & Risk Intelligence Platform is structured as two decoupled, independently scalable cloud microservices communicating over standard HTTPS:

```
                            +-------------------------------------------------------+
                            |                  End-User / Risk Analyst             |
                            +-------------------------------------------------------+
                                                       |
                                                       v [HTTPS]
+---------------------------------------------------------------------------------------------------+
|  FRONTEND SERVICE: Streamlit Risk Workbench (Port 8501)                                          |
|  - Container: Dockerfile.streamlit (Non-root user UID 1001)                                       |
|  - Modules: Executive Overview, Investigation Queue, Transaction Explorer, Model Governance,      |
|             Policy Simulator, Audit Explorer                                                      |
|  - Offline Resilience: Built-in Demo Mode (Cached benchmark metrics when API is unreachable)      |
|  - Config: FRAUD_API_URL (Dynamic HTTPS endpoint)                                                 |
+---------------------------------------------------------------------------------------------------+
                                                       |
                                                       v [HTTPS / Private Network]
+---------------------------------------------------------------------------------------------------+
|  BACKEND SERVICE: FastAPI Transaction Scoring Engine (Port 8000)                                  |
|  - Container: Dockerfile.api (Non-root user UID 1001)                                             |
|  - Engine: Calibrated XGBoost + DeepAutoencoder Anomaly + TreeSHAP Local Attributions            |
|  - Strict Validation: Pydantic v2 boundary schema, negative amounts rejected, leakage columns blocked|
|  - Security: Environment-configurable CORS, SLA timeouts, rate limiting hooks                     |
|  - Health: /health (Liveness), /ready (Readiness probe)                                           |
+---------------------------------------------------------------------------------------------------+
                                                       |
                                                       v [SQLAlchemy Connection Pool / WAL]
+---------------------------------------------------------------------------------------------------+
|  DUAL-MODE PERSISTENCE LAYER:                                                                     |
|  - Primary / Local: SQLite 3.45 in Write-Ahead Logging (PRAGMA journal_mode=WAL)                  |
|  - Production PaaS: Managed PostgreSQL 16 (Neon / Supabase / AWS RDS / Fly Postgres)              |
|  - Safety Net: Automatic fallback to local SQLite if PostgreSQL connection fails                  |
|  - Status: PostgreSQL driver integrated and verified; marked unverified in live production        |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Docker Architecture & Non-Root Security

Both container images enforce least-privilege security controls conforming to enterprise financial standards:
- **Base Image:** Minimal `python:3.11-slim` with multi-stage builds.
- **Unprivileged User:** Dedicated `appuser:appgroup` (UID `1001`, GID `1001`). No container executes as `root`.
- **Attack Surface Minimization:** Build compilers (`gcc`, `build-essential`) are purged from runtime stages.
- **Healthchecks:** Native HTTP probes (`/health` for backend, `/_stcore/health` for frontend).

### Building Docker Images Locally

```bash
# 1. Build Backend API Image
docker build -t financial-fraud-api:latest -f Dockerfile.api .

# 2. Build Frontend Dashboard Image
docker build -t financial-fraud-dashboard:latest -f Dockerfile.streamlit .
```

---

## 3. Local Multi-Container Deployment (Docker Compose)

For local end-to-end integration testing, `docker-compose.yml` orchestrates three isolated services on an internal bridge network:

```bash
# Launch full stack in background
docker compose up -d --build

# Verify container status and health
docker compose ps

# Inspect logs
docker compose logs -f fraud-api
docker compose logs -f fraud-dashboard

# Tear down stack (preserving database volume)
docker compose down
```

### Exposed Endpoints in Docker Compose:
- **FastAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **FastAPI Health Probe:** [http://localhost:8000/health](http://localhost:8000/health)
- **Streamlit Workbench:** [http://localhost:8501](http://localhost:8501)
- **PostgreSQL Database:** Internal to `fraud-net` network only (`port 5432` is not published to host).

---

## 4. Public Cloud Hosting Provider Evaluation

| Platform | Recommended Service Role | Free / Low-Cost Tier | Pros | Cons & Caveats |
| :--- | :--- | :--- | :--- | :--- |
| **Render** | Backend API & PostgreSQL | Free tier (Web Services spin down on idle; Starter \$7/mo) | Native Docker, free managed PostgreSQL, automated HTTPS, environment groups | Free web service sleeps after 15 min inactivity (50s cold start). |
| **Railway** | Backend API & Database | \$5/month usage credit (Hobby tier) | Extremely fast deploy, private internal networking, native PostgreSQL plugin | Hard resource cutoffs when usage credit is exhausted without paid card. |
| **Fly.io** | Backend API | Free allowance (up to 3 shared-cpu-1x VMs) | Global edge deployment, fast container starts, private `.internal` DNS | Requires credit card verification and CLI tooling (`flyctl`). |
| **Streamlit Community Cloud** | Frontend Dashboard | 100% Free for public GitHub repositories | Zero server management, instantaneous deploys, native Streamlit support | 1 GB memory limit per app; ephemeral filesystem (cannot host persistent SQLite). |
| **Neon / Supabase** | Serverless PostgreSQL | Free tier (0.5 GB storage, serverless compute) | Fully managed Postgres 16, connection pooling, zero server maintenance | Compute scales to zero during inactivity (cold start ~500ms on first query). |

---

## 5. Recommended Lowest-Cost Production Setup

To achieve high reliability, zero cold starts, and full separation of concerns at minimal cost:

### Strategy: Decoupled Multi-Cloud Architecture
1. **Frontend:** Deploy Streamlit Dashboard on **Streamlit Community Cloud** (Free, 0$/mo).
   - Configure Secret `FRAUD_API_URL` pointing to backend HTTPS address.
2. **Backend:** Deploy FastAPI Backend on **Render Starter Web Service** (\$7/mo) or **Railway** (\$5/mo).
   - Provides always-on compute (no 50s cold starts) and 512 MB RAM.
3. **Audit Database:** Deploy managed PostgreSQL on **Neon Free Tier** (0$/mo) or **Render Managed Postgres** (Free 90-day trial / \$7/mo).

**Total Estimated Operating Cost:** **\$5.00 – \$7.00 / month**.

---

## 6. Step-by-Step Public Deployment Guide

### Option A: Streamlit Community Cloud (Frontend) + Render (Backend)

#### Step 1: Deploy Backend on Render
1. Push this repository to GitHub.
2. Log into [Render Dashboard](https://dashboard.render.com/) and click **New + > Web Service**.
3. Connect your GitHub repository.
4. Set the following parameters:
   - **Name:** `fraud-scoring-api`
   - **Environment:** `Docker`
   - **Dockerfile Path:** `Dockerfile.api`
   - **Plan:** `Starter` (or `Free` for testing)
5. Add Environment Variables:
   - `PORT` = `8000`
   - `DATABASE_URL` = `postgresql://...` (from your Neon/Supabase/Render Postgres instance)
   - `CORS_ALLOWED_ORIGINS` = `https://share.streamlit.io,https://*.streamlit.app`
   - `LOG_LEVEL` = `INFO`
6. Click **Create Web Service**. Note the assigned URL: `https://fraud-scoring-api.onrender.com`.

#### Step 2: Deploy Frontend on Streamlit Community Cloud
1. Log into [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **New app**.
3. Select your repository, branch (`main`), and set **Main file path** to `dashboard/app.py`.
4. Click **Advanced settings... > Secrets** and enter:
   ```toml
   FRAUD_API_URL = "https://fraud-scoring-api.onrender.com"
   FRAUD_API_TIMEOUT = "10.0"
   ```
5. Click **Deploy!**.

---

## 7. Environment Configuration Reference Table

| Variable Name | Service | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `FRAUD_API_URL` | Frontend | `http://127.0.0.1:8000` | Target URL of FastAPI scoring backend. |
| `FRAUD_API_TIMEOUT` | Frontend | `5.0` | HTTP request timeout in seconds. |
| `FRAUD_DB_PATH` | Frontend | `data/audit_scoring.db` | Local SQLite fallback audit database path. |
| `PORT` | Backend | `8000` | Web server listening port. |
| `HOST` | Backend | `0.0.0.0` | Bind address. |
| `DATABASE_URL` | Backend | `sqlite:///data/audit_scoring.db` | Persistence connection string (SQLite or PostgreSQL). |
| `CORS_ALLOWED_ORIGINS` | Backend | `*` (Dev) | Comma-separated list of allowed HTTPS frontend origins. |
| `LOG_LEVEL` | Both | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `MPLCONFIGDIR` | Both | `/tmp/matplotlib` | Writable directory for headless Matplotlib cache. |

---

## 8. Ephemeral Filesystem Risk & Database Review

> [!WARNING]
> **Ephemeral Filesystem Alert (PaaS Hosting):**  
> On platforms like Render, Railway, Fly.io, and Streamlit Cloud, container local filesystems are ephemeral. Any data written to a local SQLite database (`data/audit_scoring.db`) will be erased whenever a container restarts, deploys, or sleeps.

### Mitigation: Dual-Mode Persistence Architecture
1. **Local Mode:** Uses SQLite 3.45 with Write-Ahead Logging (`PRAGMA journal_mode=WAL`), suitable for offline developer evaluation and local Docker runs.
2. **Cloud Mode:** Set `DATABASE_URL=postgresql://user:pass@host:5432/dbname`. The backend connects via SQLAlchemy connection pooling with keepalive probes (`pool_pre_ping=True`).
3. **Resilience Fallback:** If PostgreSQL credentials expire or network partitions occur, `DatabaseManager` logs a structured warning and falls back to local SQLite, preventing transaction rejection.

---

## 9. Security & Compliance Checklist

- [x] **Non-Root Execution:** All Docker containers run as unprivileged `appuser` (UID 1001).
- [x] **No Hardcoded Secrets:** Passwords and tokens are ingested strictly via environment variables.
- [x] **Safe CORS Configuration:** FastAPI restricts cross-origin scoring requests to authorized domains via `CORS_ALLOWED_ORIGINS`.
- [x] **PII Account Masking:** Customer accounts are masked at presentation and logging boundaries (`C1****7890`).
- [x] **Safe Database URLs:** Database connection URLs containing passwords are automatically sanitized (`sanitize_db_url`) before writing to application logs.
- [x] **Input Range Validation:** Transactions with negative amounts, negative balances, or invalid transfer types are blocked at HTTP schema validation before reaching inference models.
- [x] **Leakage Columns Blocked:** Attempts to pass target labels (`isFraud`, `newbalanceOrig`, `isFlaggedFraud`) are rejected with `HTTP 422`.
- [x] **Demo Mode Offline Fallback:** Streamlit frontend displays clear operational banners when backend is offline without fabricating mock scores.
- [x] **Compliance Clarification:** Platform is a research prototype on synthetic data; account masking adheres to PCI-DSS best practices without claiming formal statutory certification.


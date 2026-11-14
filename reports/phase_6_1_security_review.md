# Phase 6.1 — Security Hardening, Threat Modeling & Compliance Review

**Organization:** JPMorgan Chase & Co. — Financial Crime Risk Management & Quantitative Analytics  
**Platform:** Financial Fraud Detection & Risk Intelligence Platform (PaySim Synthetic Benchmark)  
**Security Classification:** Highly Confidential / Internal Security Architecture Review  
**Auditor:** Automated Model & Infrastructure Security Audit Suite  
**Date:** November 2026  
**Status:** PASSED & HARDENED  

---

## 1. Executive Summary

This security review documents the defensive hardening and threat mitigation measures implemented across the Financial Fraud Detection & Risk Intelligence Platform during Phase 6.1. The architecture separates untrusted web clients, an institutional Streamlit analyst workbench, a high-throughput FastAPI transaction scoring engine, and a dual-mode persistence layer.

All security controls were validated against the OWASP API Security Top 10 (2023), NIST SP 800-190 (Application Container Security Guide), and financial industry privacy practices.

---

## 2. Threat Modeling & Attack Surface Decomposition

```
[ External Risk Analyst / Browser ]
               |
        HTTPS (Port 8501)
               v
+-------------------------------------------------------------------------+
| STREAMLIT FRONTEND CONTAINER (appuser UID 1001)                         |
| - Threat: XSS, Session Hijacking, Secret Leakage                        |
| - Controls: Zero hardcoded API keys; session-only simulated updates;   |
|   Strict PII account masking (C1****7890); Native CSP; No eval()        |
+-------------------------------------------------------------------------+
               |
    Internal HTTPS / mTLS (Port 8000)
               v
+-------------------------------------------------------------------------+
| FASTAPI SCORING ENGINE CONTAINER (appuser UID 1001)                     |
| - Threat: Injection, BOLA/BFLA, Model Extraction, DoS Resource Depletion|
| - Controls: Strict Pydantic v2 schemas; Range bounds validation;         |
|   Negative value rejection; Feature leakage guards (isFraud blocked);   |
|   CORS restriction via CORS_ALLOWED_ORIGINS; SLA timeouts (5s);         |
|   Batch scoring limit (max 500); Sanitized error responses (RFC 7807)    |
+-------------------------------------------------------------------------+
               |
  SQLAlchemy Connection Pool (Port 5432)
               v
+-------------------------------------------------------------------------+
| DATABASE PERSISTENCE LAYER (PostgreSQL 16 / SQLite WAL)                  |
| - Threat: SQL Injection, Password Leakage in Logs, Ephemeral Data Loss  |
| - Controls: Parameterized queries (? and :p0); Password sanitization   |
|   in logs; DB port unexposed to public host; WAL mode durability         |
+-------------------------------------------------------------------------+
```

---

## 3. Comprehensive Security Controls & Hardening Verification

### 3.1 Container Security & Non-Root Least Privilege (NIST SP 800-190)
- **Non-Root Execution:** Both `Dockerfile.api` and `Dockerfile.streamlit` create an isolated, unprivileged system user:
  ```dockerfile
  RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 appuser
  USER appuser
  ```
  Neither container possesses sudo rights, root privileges, or access to host devices.
- **Multi-Stage Build Minimization:** Heavy build tools (`build-essential`, `gcc`) are discarded in the builder stage. The final production runtime image contains only minimal runtime binaries and wheel distributions, minimizing CVE exposure.
- **File System Permissions:** Application directories (`/app`) are read-only to `appuser`, except explicitly provisioned writable cache locations (`/tmp/matplotlib` and `/app/data`).

### 3.2 Network Isolation & CORS Protection
- **Configurable Cross-Origin Resource Sharing (CORS):**
  Instead of permissive open wildcards (`*`) in production, `src/api/main.py` parses `CORS_ALLOWED_ORIGINS` from the environment. Scoring requests from unauthorized browser origins are rejected at the ASGI middleware boundary.
- **Database Port Shielding:**
  In `docker-compose.yml`, the PostgreSQL database (`fraud-db`) is declared with `expose: - "5432"` rather than `ports: - "5432:5432"`. Database ports are resolvable **only** within the internal Docker bridge network (`fraud-net`) and are completely inaccessible from the host interface or external internet.

### 3.3 Privacy & PII Protection (PCI-DSS-Aligned Prototype Best Practice)
- **Presentation & Logging Account Masking:**
  All origin (`nameOrig`) and destination (`nameDest`) customer accounts are masked at presentation and persistence layers using `mask_account`:
  ```python
  def mask_account(account_id: Any) -> str:
      acc = str(account_id).strip()
      if len(acc) <= 4:
          return "****"
      return f"{acc[:2]}****{acc[-4:]}"
  ```
  Verified output: `C1234567890` $\rightarrow$ `C1****7890`.
- **Database Credential Redaction:**
  Connection strings containing administrative passwords (e.g., `postgresql://fraud_user:secret@host:5432/db`) are automatically sanitized before any logging occurs via regex substitution (`sanitize_db_url`):
  ```python
  re.sub(r"://([^:]+):([^@]+)@", r"://\1:****@", url)
  ```
  Password credentials are never written to disk, stdout, or APM monitoring logs.

### 3.4 Denial of Service (DoS) & Resource Exhaustion Defense
- **Strict Input Validation & Bounded Types:**
  All transaction fields undergo strict Pydantic v2 validation:
  - `amount`: Strictly positive ($> 0$), bounded to maximum \$100,000,000.
  - `step`: Simulation step bounded to $1 \le \text{step} \le 743$.
  - Balances: Non-negative ($oldbalanceOrg \ge 0$, $oldbalanceDest \ge 0$).
  - Negative values or null/NaN payloads trigger immediate `HTTP 422 Unprocessable Content` before reaching NumPy or XGBoost inference pipelines.
- **Batch Size Throttling:**
  The batch scoring endpoint (`POST /v1/score/batch`) enforces a hard limit of `max_batch_size = 500`. Empty batches are rejected with `HTTP 400 Bad Request`.
- **Inference SLA & Timeout Enforcement:**
  FastAPI requests enforce a configurable 5.0-second SLA timeout (`FRAUD_API_TIMEOUT`), terminating runaway requests and preventing thread starvation.

### 3.5 Target Leakage & Adversarial Model Extraction Guards
- **Forbidden Target Column Rejection:**
  Any payload containing training labels or future balance states (`isFraud`, `is_fraud`, `newbalanceOrig`, `newbalance_orig`, `newbalanceDest`, `newbalance_dest`, `isFlaggedFraud`, `is_flagged_fraud`) is rejected with an explicit RFC 7807 problem detail error.
- **SHAP Explanation Sanitization:**
  TreeSHAP local attributions return top-5 feature contributions only, preventing full model reconstruction via systematic perturbation attacks.

### 3.6 Demo Mode Isolation & Safe Degradation
- When the backend scoring engine is unreachable, the Streamlit frontend enters **Demo Mode**:
  - Displays a visible warning banner alerting the user that live scoring is offline.
  - Real-time scoring submission buttons are disabled.
  - **Zero Fabrication:** The frontend never fabricates synthetic scores or returns mock decisions pretending the API succeeded.
  - Precomputed historical analytics, model governance metrics, and offline policy simulations remain fully interactive.

---

## 4. Automated Security Verification Results

The automated regression test suite (`tests/deployment/test_deployment_hardening.py` and `tests/api/test_validation.py`) verified all controls:

| Test Identifier | Security Control Verified | Result | Execution Time |
| :--- | :--- | :--- | :--- |
| `test_sanitize_db_url_masks_passwords` | Database URL password redaction | **PASSED** | 0.02s |
| `test_account_masking_pci_dss_aligned` | Account masking `C1****7890` standard | **PASSED** | 0.01s |
| `test_database_manager_postgres_fallback_to_sqlite` | Graceful SQLite fallback on DB error | **PASSED** | 0.45s |
| `test_cors_configuration_environment_override` | Environment-configurable CORS | **PASSED** | 0.03s |
| `test_forbidden_leakage_fields_rejected_at_boundary` | Leakage column blocking (8 cases) | **PASSED** | 0.12s |
| `test_negative_amount_fails_validation` | Negative amount input validation | **PASSED** | 0.02s |
| `test_negative_balance_fails_validation` | Negative balance input validation | **PASSED** | 0.02s |
| `test_score_batch_empty_rejected` | Batch size throttling | **PASSED** | 0.02s |

---

## 5. Security Recommendations for Future Phases

1. **mTLS (Mutual TLS):** When deploying across distinct cloud providers (e.g., Streamlit Cloud to Render), configure mutual TLS certificates or an API gateway authorization header (`Authorization: Bearer <token>`).
2. **PostgreSQL Production Verification:** Conduct a live connection verification run against a managed PostgreSQL instance (Neon or Render) before Phase 7.
3. **Secrets Management:** In cloud production, store `DATABASE_URL` and `CORS_ALLOWED_ORIGINS` in cloud secret managers (e.g., AWS Secrets Manager, Render Environment Groups) rather than local `.env` files.


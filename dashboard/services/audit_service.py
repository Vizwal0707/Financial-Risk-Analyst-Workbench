"""Read-only audit repository service for the Streamlit dashboard."""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from dashboard.config.settings import settings
from dashboard.utils.formatters import mask_account

logger = logging.getLogger(__name__)


class AuditDataService:
    """Provides read-only access to persisted SQLite scoring audit logs."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or settings.database_path

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        """Creates a read-only SQLite connection."""
        if not os.path.exists(self.db_path):
            return None
        try:
            # Open with URI read-only flag
            uri_path = f"file:{os.path.abspath(self.db_path)}?mode=ro"
            conn = sqlite3.connect(uri_path, uri=True)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.warning(f"Failed to open SQLite database in read-only mode, falling back to standard connect: {e}")
            try:
                conn = sqlite3.connect(self.db_path)
                conn.row_factory = sqlite3.Row
                return conn
            except Exception as ex:
                logger.error(f"Cannot connect to audit database at {self.db_path}: {ex}")
                return None

    def get_audit_summary(self) -> Dict[str, Any]:
        """Calculates aggregate portfolio summary from audit records."""
        conn = self._get_connection()
        if conn is None:
            return {
                "total_count": 0,
                "approve_count": 0,
                "review_count": 0,
                "decline_count": 0,
                "total_amount": 0.0,
                "declined_amount": 0.0,
                "mean_fraud_prob": 0.0,
                "mean_latency": 0.0,
                "status": "UNAVAILABLE",
            }

        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT
                    COUNT(*) AS total_count,
                    SUM(CASE WHEN decision = 'APPROVE' THEN 1 ELSE 0 END) AS approve_count,
                    SUM(CASE WHEN decision = 'REVIEW' THEN 1 ELSE 0 END) AS review_count,
                    SUM(CASE WHEN decision = 'DECLINE' THEN 1 ELSE 0 END) AS decline_count,
                    COALESCE(SUM(amount), 0.0) AS total_amount,
                    COALESCE(SUM(CASE WHEN decision = 'DECLINE' THEN amount ELSE 0.0 END), 0.0) AS declined_amount,
                    COALESCE(AVG(calibrated_probability), 0.0) AS mean_fraud_prob,
                    COALESCE(AVG(latency_ms), 0.0) AS mean_latency
                FROM audit_scoring_logs;
            """)
            row = cur.fetchone()
            if not row or row["total_count"] == 0:
                return {
                    "total_count": 0,
                    "approve_count": 0,
                    "review_count": 0,
                    "decline_count": 0,
                    "total_amount": 0.0,
                    "declined_amount": 0.0,
                    "mean_fraud_prob": 0.0,
                    "mean_latency": 0.0,
                    "status": "EMPTY",
                }

            return {
                "total_count": int(row["total_count"]),
                "approve_count": int(row["approve_count"] or 0),
                "review_count": int(row["review_count"] or 0),
                "decline_count": int(row["decline_count"] or 0),
                "total_amount": float(row["total_amount"]),
                "declined_amount": float(row["declined_amount"]),
                "mean_fraud_prob": float(row["mean_fraud_prob"]),
                "mean_latency": float(row["mean_latency"]),
                "status": "ACTIVE_SQLITE_WAL",
            }
        except Exception as e:
            logger.error(f"Error querying audit summary: {e}")
            return {"status": "ERROR", "error": str(e), "total_count": 0}
        finally:
            conn.close()

    def get_recent_logs(self, limit: int = 1000) -> pd.DataFrame:
        """Loads recent audit records into a pandas DataFrame."""
        conn = self._get_connection()
        if conn is None:
            return pd.DataFrame()

        try:
            sql = f"""
                SELECT
                    id, request_id, step, corridor_type, amount, name_orig, oldbalance_orig,
                    name_dest, oldbalance_dest, raw_probability, calibrated_probability,
                    risk_score, risk_tier, decision, action_code, decision_reasons,
                    anomaly_score, latency_ms, model_version, cold_start_applied, created_at
                FROM audit_scoring_logs
                ORDER BY id DESC
                LIMIT {limit};
            """
            df = pd.read_sql_query(sql, conn)
            if df.empty:
                return df

            # Apply PII masking to accounts
            df["name_orig_masked"] = df["name_orig"].apply(mask_account)
            df["name_dest_masked"] = df["name_dest"].apply(mask_account)

            # Compute risk_weighted_investigation_priority = P_cal * ln(1 + amount)
            df["risk_weighted_investigation_priority"] = (
                df["calibrated_probability"] * np.log1p(df["amount"].clip(lower=0.0))
            ).round(4)

            # Parse JSON decision_reasons string
            def _parse_reasons(r):
                if not r:
                    return []
                try:
                    return json.loads(r)
                except Exception:
                    return [r]

            df["parsed_reasons"] = df["decision_reasons"].apply(_parse_reasons)
            return df
        except Exception as e:
            logger.error(f"Error reading recent logs: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def get_record_by_request_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves single audit record with full explanation JSON."""
        conn = self._get_connection()
        if conn is None:
            return None

        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT * FROM audit_scoring_logs WHERE request_id = ? LIMIT 1;
            """, (request_id,))
            row = cur.fetchone()
            if not row:
                return None

            record = dict(row)
            record["name_orig_masked"] = mask_account(record.get("name_orig"))
            record["name_dest_masked"] = mask_account(record.get("name_dest"))

            # Calculate priority
            p_cal = float(record.get("calibrated_probability") or 0.0)
            amt = float(record.get("amount") or 0.0)
            record["risk_weighted_investigation_priority"] = round(p_cal * np.log1p(max(amt, 0.0)), 4)

            # Parse reasons
            if record.get("decision_reasons"):
                try:
                    record["decision_reasons"] = json.loads(record["decision_reasons"])
                except Exception:
                    record["decision_reasons"] = [record["decision_reasons"]]
            else:
                record["decision_reasons"] = []

            # Parse explanation JSON
            if record.get("explanation_json"):
                try:
                    record["explanation"] = json.loads(record["explanation_json"])
                except Exception:
                    record["explanation"] = None
            else:
                record["explanation"] = None

            return record
        except Exception as e:
            logger.error(f"Error retrieving audit record {request_id}: {e}")
            return None
        finally:
            conn.close()


audit_service = AuditDataService()


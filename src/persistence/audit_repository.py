"""Audit repository for regulatory persistence, idempotency lookup, and transaction logs."""

import json
import logging
from typing import Any, Dict, List, Optional
from src.persistence.database import DatabaseManager

logger = logging.getLogger(__name__)


class AuditRepository:
    """Repository managing asynchronous audit persistence and idempotent scoring retrieval."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    async def get_by_request_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves existing scoring decision by request_id for idempotency deduplication."""
        sql = """
        SELECT request_id, step, corridor_type, amount, name_orig, oldbalance_orig,
               name_dest, oldbalance_dest, raw_probability, calibrated_probability,
               risk_score, risk_tier, decision, action_code, decision_reasons,
               anomaly_score, latency_ms, model_version, cold_start_applied,
               created_at, explanation_json
        FROM audit_scoring_logs
        WHERE request_id = ?
        LIMIT 1;
        """
        rows = await self.db.execute_query(sql, (request_id,))
        if not rows:
            return None

        record = rows[0]
        record["operational_recommendation"] = record.get("action_code")
        # Deserialize JSON reason codes and explanation
        if record.get("decision_reasons"):
            try:
                record["decision_reasons"] = json.loads(record["decision_reasons"])
            except Exception:
                record["decision_reasons"] = [record["decision_reasons"]]
        else:
            record["decision_reasons"] = []

        if record.get("explanation_json"):
            try:
                record["explanation"] = json.loads(record["explanation_json"])
            except Exception:
                record["explanation"] = None
        else:
            record["explanation"] = None

        return record

    async def log_decision(
        self,
        request_id: str,
        step: int,
        corridor_type: str,
        amount: float,
        name_orig: str,
        oldbalance_orig: float,
        name_dest: str,
        oldbalance_dest: float,
        raw_probability: float,
        calibrated_probability: float,
        risk_score: float,
        risk_tier: str,
        decision: str,
        action_code: str,
        decision_reasons: List[str],
        latency_ms: float,
        model_version: str,
        anomaly_score: Optional[float] = None,
        cold_start_applied: bool = False,
        explanation_json: Optional[str] = None,
    ) -> None:
        """Persists transaction decision audit record."""
        sql = """
        INSERT INTO audit_scoring_logs (
            request_id, step, corridor_type, amount, name_orig, oldbalance_orig,
            name_dest, oldbalance_dest, raw_probability, calibrated_probability,
            risk_score, risk_tier, decision, action_code, decision_reasons,
            anomaly_score, latency_ms, model_version, cold_start_applied,
            explanation_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (request_id) DO NOTHING;
        """
        reasons_str = json.dumps(decision_reasons)
        params = (
            request_id,
            int(step),
            str(corridor_type),
            float(amount),
            str(name_orig),
            float(oldbalance_orig),
            str(name_dest),
            float(oldbalance_dest),
            float(raw_probability),
            float(calibrated_probability),
            float(risk_score),
            str(risk_tier),
            str(decision),
            str(action_code),
            reasons_str,
            float(anomaly_score) if anomaly_score is not None else None,
            float(latency_ms),
            str(model_version),
            1 if cold_start_applied else 0,
            explanation_json,
        )
        try:
            await self.db.execute_mutation(sql, params)
        except Exception as e:
            # Catch duplicate key errors gracefully
            logger.warning(f"Failed to insert audit record for request_id={request_id}: {e}")

    async def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves recent scoring audit entries for analyst monitoring."""
        sql = """
        SELECT request_id, step, corridor_type, amount, name_orig, name_dest,
               raw_probability, calibrated_probability, risk_score, risk_tier,
               decision, action_code, decision_reasons, anomaly_score, latency_ms,
               created_at
        FROM audit_scoring_logs
        ORDER BY id DESC
        LIMIT ?;
        """
        return await self.db.execute_query(sql, (limit,))

    async def get_audit_summary(self) -> Dict[str, Any]:
        """Calculates aggregate portfolio metrics from audit logs."""
        sql = """
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
        """
        rows = await self.db.execute_query(sql)
        if not rows:
            return {
                "total_count": 0,
                "approve_count": 0,
                "review_count": 0,
                "decline_count": 0,
                "total_amount": 0.0,
                "declined_amount": 0.0,
                "mean_fraud_prob": 0.0,
                "mean_latency": 0.0,
            }
        return rows[0]

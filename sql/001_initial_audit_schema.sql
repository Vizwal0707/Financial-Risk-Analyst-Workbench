-- =============================================================================
-- Migration: 001_initial_audit_schema.sql
-- Description: Audit persistence table for transaction scoring decisions,
--              calibrated probabilities, triage routing, and explainability cards.
-- Compatible with PostgreSQL 16 and SQLite 3.
-- =============================================================================

CREATE TABLE IF NOT EXISTS audit_scoring_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id VARCHAR(64) NOT NULL UNIQUE,
    step INTEGER NOT NULL,
    corridor_type VARCHAR(16) NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    name_orig VARCHAR(64) NOT NULL,
    oldbalance_orig DOUBLE PRECISION NOT NULL,
    name_dest VARCHAR(64) NOT NULL,
    oldbalance_dest DOUBLE PRECISION NOT NULL,
    raw_probability DOUBLE PRECISION NOT NULL,
    calibrated_probability DOUBLE PRECISION NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    risk_tier VARCHAR(16) NOT NULL,
    decision VARCHAR(16) NOT NULL,
    action_code VARCHAR(32) NOT NULL,
    decision_reasons TEXT,
    anomaly_score DOUBLE PRECISION,
    latency_ms DOUBLE PRECISION NOT NULL,
    model_version VARCHAR(32) NOT NULL,
    cold_start_applied INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    explanation_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_audit_request_id ON audit_scoring_logs (request_id);
CREATE INDEX IF NOT EXISTS idx_audit_decision ON audit_scoring_logs (decision);
CREATE INDEX IF NOT EXISTS idx_audit_created_at ON audit_scoring_logs (created_at);
CREATE INDEX IF NOT EXISTS idx_audit_name_orig ON audit_scoring_logs (name_orig);


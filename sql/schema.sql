-- Financial Fraud Detection & Risk Intelligence Platform
-- PostgreSQL Operational Database Schema (DDL)
-- Version: 1.0.0

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Ingested Transactions Store
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(64) PRIMARY KEY,
    step INTEGER NOT NULL,
    type VARCHAR(32) NOT NULL,
    amount NUMERIC(18, 4) NOT NULL,
    name_orig VARCHAR(64) NOT NULL,
    oldbalance_org NUMERIC(18, 4) NOT NULL,
    name_dest VARCHAR(64) NOT NULL,
    oldbalance_dest NUMERIC(18, 4) NOT NULL,
    is_fraud_ground_truth SMALLINT DEFAULT NULL, -- Nullable for unlabelled live/test events
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transactions_step ON transactions(step);
CREATE INDEX IF NOT EXISTS idx_transactions_name_orig ON transactions(name_orig);
CREATE INDEX IF NOT EXISTS idx_transactions_name_dest ON transactions(name_dest);
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(type);

-- 2. Real-time Risk Assessments & Model Scoring Audit
CREATE TABLE IF NOT EXISTS risk_assessments (
    assessment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id VARCHAR(64) NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    raw_xgb_probability NUMERIC(8, 6) NOT NULL,
    calibrated_xgb_probability NUMERIC(8, 6) NOT NULL,
    calibration_method VARCHAR(32) NOT NULL, -- 'isotonic' or 'sigmoid'
    ae_reconstruction_error NUMERIC(12, 6) NOT NULL,
    ae_calibrated_score NUMERIC(8, 6) NOT NULL,
    hybrid_risk_score NUMERIC(8, 6) NOT NULL,
    hybrid_alpha NUMERIC(4, 2) NOT NULL,
    assigned_tier VARCHAR(16) NOT NULL, -- 'APPROVE', 'REVIEW', 'DECLINE'
    scored_in_ms NUMERIC(8, 3) NOT NULL,
    scored_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_assessments_transaction_id ON risk_assessments(transaction_id);
CREATE INDEX IF NOT EXISTS idx_assessments_tier ON risk_assessments(assigned_tier);
CREATE INDEX IF NOT EXISTS idx_assessments_hybrid_score ON risk_assessments(hybrid_risk_score);
CREATE INDEX IF NOT EXISTS idx_assessments_scored_at ON risk_assessments(scored_at);

-- 3. Explainability & SHAP Attributions Store
CREATE TABLE IF NOT EXISTS shap_attributions (
    attribution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    assessment_id UUID NOT NULL REFERENCES risk_assessments(assessment_id) ON DELETE CASCADE,
    top_feature_1_name VARCHAR(64) NOT NULL,
    top_feature_1_val NUMERIC(12, 4) NOT NULL,
    top_feature_1_shap NUMERIC(10, 6) NOT NULL,
    top_feature_2_name VARCHAR(64) NOT NULL,
    top_feature_2_val NUMERIC(12, 4) NOT NULL,
    top_feature_2_shap NUMERIC(10, 6) NOT NULL,
    top_feature_3_name VARCHAR(64) NOT NULL,
    top_feature_3_val NUMERIC(12, 4) NOT NULL,
    top_feature_3_shap NUMERIC(10, 6) NOT NULL,
    explanation_summary TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_shap_assessment_id ON shap_attributions(assessment_id);

-- 4. Level-1 Investigation Cases & Operational Queue
CREATE TABLE IF NOT EXISTS investigation_cases (
    case_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    assessment_id UUID NOT NULL REFERENCES risk_assessments(assessment_id) ON DELETE CASCADE,
    transaction_id VARCHAR(64) NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    status VARCHAR(32) NOT NULL DEFAULT 'QUEUED', -- 'QUEUED', 'IN_REVIEW', 'CONFIRMED_FRAUD', 'FALSE_POSITIVE'
    priority VARCHAR(16) NOT NULL DEFAULT 'NORMAL', -- 'LOW', 'NORMAL', 'HIGH', 'CRITICAL'
    assigned_analyst VARCHAR(64) DEFAULT NULL,
    analyst_decision VARCHAR(32) DEFAULT NULL, -- 'SAR_FILED', 'ACCOUNT_FROZEN', 'FALSE_ALARM_DISMISSED'
    investigation_notes TEXT DEFAULT NULL,
    queued_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP WITH TIME ZONE DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_cases_status ON investigation_cases(status);
CREATE INDEX IF NOT EXISTS idx_cases_priority ON investigation_cases(priority);
CREATE INDEX IF NOT EXISTS idx_cases_assigned_analyst ON investigation_cases(assigned_analyst);

-- 5. Daily / Hourly Model Performance & Drift Audit Log
CREATE TABLE IF NOT EXISTS model_drift_audit (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    audit_step_start INTEGER NOT NULL,
    audit_step_end INTEGER NOT NULL,
    psi_amount NUMERIC(8, 4) NOT NULL,
    psi_hybrid_score NUMERIC(8, 4) NOT NULL,
    ks_statistic_amount NUMERIC(8, 4) NOT NULL,
    ks_pvalue_amount NUMERIC(10, 6) NOT NULL,
    alerts_triggered INTEGER NOT NULL DEFAULT 0,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


-- Financial Fraud Detection & Risk Intelligence Platform
-- High-Performance Analytical Queries (DuckDB & PostgreSQL Compatible)
-- Version: 1.0.0

-- Query 1: Overall Portfolio Summary & Class Distribution by Corridor
SELECT 
    type AS transaction_type,
    COUNT(*) AS total_transactions,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(amount), 2) AS total_volume,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END), 2) AS total_fraud_volume,
    ROUND(AVG(amount), 2) AS avg_transaction_amount,
    ROUND(MAX(amount), 2) AS max_transaction_amount
FROM read_parquet('data/processed/paysim_featurized.parquet')
GROUP BY type
ORDER BY fraud_transactions DESC;

-- Query 2: Hourly Fraud Velocity & Peak Activity Windows
SELECT 
    (step % 24) AS hour_of_day,
    COUNT(*) AS total_volume,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END), 2) AS fraud_exposure_amount
FROM read_parquet('data/processed/paysim_featurized.parquet')
GROUP BY (step % 24)
ORDER BY hour_of_day;

-- Query 3: Cumulative Step-by-Step Fraud Value vs Transaction Count
SELECT 
    step,
    COUNT(*) AS transactions_at_step,
    SUM(amount) AS volume_at_step,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_count_at_step,
    SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END) AS fraud_amount_at_step,
    SUM(SUM(CASE WHEN isFraud = 1 THEN amount ELSE 0 END)) OVER (ORDER BY step) AS cumulative_fraud_loss
FROM read_parquet('data/processed/paysim_featurized.parquet')
GROUP BY step
ORDER BY step;

-- Query 4: Zero-Balance Origination Anomaly Check
SELECT 
    is_orig_zero_balance,
    type,
    COUNT(*) AS tx_count,
    SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
    ROUND(SUM(CASE WHEN isFraud = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) AS fraud_rate_pct,
    ROUND(AVG(amount), 2) AS avg_amount
FROM read_parquet('data/processed/paysim_featurized.parquet')
GROUP BY is_orig_zero_balance, type
ORDER BY fraud_rate_pct DESC;

-- Query 5: Operational Investigation Queue Backlog (PostgreSQL)
SELECT 
    status,
    priority,
    COUNT(*) AS case_count,
    MIN(queued_at) AS oldest_pending_case,
    MAX(queued_at) AS newest_case
FROM investigation_cases
GROUP BY status, priority
ORDER BY priority DESC, case_count DESC;


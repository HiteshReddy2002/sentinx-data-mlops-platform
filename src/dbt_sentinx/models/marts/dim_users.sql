WITH users AS (
    SELECT * FROM {{ source('bronze', 'raw_users') }}
),

user_tx_aggregates AS (
    SELECT
        user_id,
        COUNT(transaction_id) AS total_transactions,
        SUM(amount) AS total_lifetime_spend,
        AVG(amount) AS avg_transaction_spend,
        SUM(is_fraud) AS fraud_incident_count,
        MAX(transaction_timestamp) AS last_active_at
    FROM {{ ref('fct_transactions') }}
    GROUP BY user_id
)

SELECT
    u.user_id,
    u.user_name,
    u.email,
    u.country,
    u.account_age_days,
    u.kyc_verified,
    u.baseline_spend,
    u.created_at,
    COALESCE(a.total_transactions, 0) AS total_transactions,
    COALESCE(ROUND(a.total_lifetime_spend, 2), 0.0) AS total_lifetime_spend,
    COALESCE(ROUND(a.avg_transaction_spend, 2), 0.0) AS avg_transaction_spend,
    COALESCE(a.fraud_incident_count, 0) AS fraud_incident_count,
    a.last_active_at,
    CASE 
        WHEN u.account_age_days < 30 THEN 'New (<30d)'
        WHEN u.account_age_days BETWEEN 30 AND 180 THEN 'Established (1-6m)'
        WHEN u.account_age_days BETWEEN 181 AND 365 THEN 'Mature (6-12m)'
        ELSE 'Veteran (>1yr)'
    END AS user_cohort_tenure
FROM users u
LEFT JOIN user_tx_aggregates a ON u.user_id = a.user_id

WITH stg_tx AS (
    SELECT * FROM {{ ref('stg_transactions') }}
),

velocity AS (
    SELECT * FROM {{ ref('int_user_transaction_velocity') }}
)

SELECT
    t.transaction_id,
    t.user_id,
    t.merchant_id,
    t.amount,
    t.currency,
    t.transaction_timestamp,
    t.transaction_date,
    t.channel,
    t.card_type,
    t.device_type,
    t.transaction_country,
    t.distance_from_home_km,
    t.hour_of_day,
    t.is_weekend,
    v.tx_count_last_1h,
    v.total_spend_last_24h,
    v.seconds_since_prev_tx,
    t.is_fraud,
    t.fraud_type
FROM stg_tx t
LEFT JOIN velocity v ON t.transaction_id = v.transaction_id

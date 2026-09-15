WITH stg_tx AS (
    SELECT * FROM {{ ref('stg_transactions') }}
),

windowed_velocity AS (
    SELECT
        transaction_id,
        user_id,
        amount,
        transaction_timestamp,
        -- Rolling 1-hour transaction count
        COUNT(transaction_id) OVER (
            PARTITION BY user_id 
            ORDER BY transaction_timestamp 
            RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW
        ) AS tx_count_last_1h,
        -- Rolling 24-hour total spend
        SUM(amount) OVER (
            PARTITION BY user_id 
            ORDER BY transaction_timestamp 
            RANGE BETWEEN INTERVAL 24 HOUR PRECEDING AND CURRENT ROW
        ) AS total_spend_last_24h,
        -- Lag previous transaction timestamp to detect velocity bursts
        LAG(transaction_timestamp) OVER (
            PARTITION BY user_id 
            ORDER BY transaction_timestamp
        ) AS prev_tx_timestamp
    FROM stg_tx
)

SELECT
    *,
    EPOCH(transaction_timestamp) - EPOCH(prev_tx_timestamp) AS seconds_since_prev_tx
FROM windowed_velocity

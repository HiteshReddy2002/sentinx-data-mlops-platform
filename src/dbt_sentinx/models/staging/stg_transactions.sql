WITH source AS (
    SELECT * FROM {{ source('bronze', 'raw_transactions') }}
),

cleaned AS (
    SELECT
        transaction_id,
        user_id,
        merchant_id,
        CAST(amount AS DOUBLE) AS amount,
        currency,
        CAST(timestamp AS TIMESTAMP) AS transaction_timestamp,
        channel,
        card_type,
        device_type,
        country AS transaction_country,
        CAST(distance_from_home_km AS DOUBLE) AS distance_from_home_km,
        CAST(is_fraud AS INTEGER) AS is_fraud,
        fraud_type,
        -- Date partition & temporal signals
        DATE_TRUNC('day', CAST(timestamp AS TIMESTAMP)) AS transaction_date,
        EXTRACT(hour FROM CAST(timestamp AS TIMESTAMP)) AS hour_of_day,
        EXTRACT(dow FROM CAST(timestamp AS TIMESTAMP)) IN (0, 6) AS is_weekend
    FROM source
    WHERE transaction_id IS NOT NULL
      AND amount > 0
)

SELECT * FROM cleaned

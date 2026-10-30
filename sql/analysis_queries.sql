-- name: target_rate_by_income_type
SELECT NAME_INCOME_TYPE, COUNT(*) AS n, AVG(TARGET) AS target_rate FROM application_train GROUP BY 1 ORDER BY n DESC;
-- name: target_rate_by_contract_type
SELECT NAME_CONTRACT_TYPE, COUNT(*) AS n, AVG(TARGET) AS target_rate FROM application_train GROUP BY 1 ORDER BY n DESC;
-- name: target_rate_by_age_band
SELECT CASE WHEN -DAYS_BIRTH/365.25 < 25 THEN '<25' WHEN -DAYS_BIRTH/365.25 < 35 THEN '25-34'
            WHEN -DAYS_BIRTH/365.25 < 45 THEN '35-44' WHEN -DAYS_BIRTH/365.25 < 55 THEN '45-54' ELSE '55+' END AS age_band,
       COUNT(*) AS n, AVG(TARGET) AS target_rate
FROM application_train GROUP BY 1 ORDER BY 1;
-- name: prior_refusals_vs_target
WITH r AS (SELECT SK_ID_CURR, SUM(CASE WHEN NAME_CONTRACT_STATUS='Refused' THEN 1 ELSE 0 END) AS refusals FROM previous_application GROUP BY 1)
SELECT LEAST(COALESCE(r.refusals,0), 5) AS prior_refusals_capped_at_5, COUNT(*) AS n, AVG(a.TARGET) AS target_rate
FROM application_train a LEFT JOIN r USING (SK_ID_CURR) GROUP BY 1 ORDER BY 1;
-- name: bureau_history_vs_target
WITH b AS (SELECT SK_ID_CURR, COUNT(*) AS n_credits FROM bureau GROUP BY 1)
SELECT CASE WHEN b.n_credits IS NULL THEN 'no bureau record' WHEN b.n_credits < 5 THEN '1-4' WHEN b.n_credits < 10 THEN '5-9' ELSE '10+' END AS bureau_credits,
       COUNT(*) AS n, AVG(a.TARGET) AS target_rate
FROM application_train a LEFT JOIN b USING (SK_ID_CURR) GROUP BY 1 ORDER BY n DESC;
-- name: key_integrity_prev_to_application
SELECT COUNT(*) AS prev_rows, COUNT(DISTINCT p.SK_ID_CURR) AS distinct_curr,
       SUM(CASE WHEN a.SK_ID_CURR IS NULL AND t.SK_ID_CURR IS NULL THEN 1 ELSE 0 END) AS orphan_rows
FROM previous_application p LEFT JOIN application_train a USING (SK_ID_CURR) LEFT JOIN application_test t USING (SK_ID_CURR);

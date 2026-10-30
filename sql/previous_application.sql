-- Previous applications -> one row per SK_ID_CURR. DAYS_DECISION > 0 (after application) is excluded.
SELECT SK_ID_CURR,
  COUNT(*)                                                           AS PREV_APP_COUNT,
  SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 ELSE 0 END) AS PREV_APPROVED_COUNT,
  SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused'  THEN 1 ELSE 0 END) AS PREVIOUS_REJECTION_COUNT,
  SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Canceled' THEN 1 ELSE 0 END) AS PREV_CANCELED_COUNT,
  AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1.0 ELSE 0.0 END) AS PREVIOUS_APPROVAL_RATIO,
  AVG(AMT_APPLICATION)                                               AS PREV_AMT_APPLICATION_MEAN,
  AVG(AMT_CREDIT)                                                    AS PREV_AMT_CREDIT_MEAN,
  AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))                       AS PREV_CREDIT_APPLICATION_RATIO,
  MAX(DAYS_DECISION)                                                 AS PREV_DAYS_DECISION_MAX,
  MIN(DAYS_DECISION)                                                 AS PREV_DAYS_DECISION_MIN,
  AVG(CNT_PAYMENT)                                                   AS PREV_CNT_PAYMENT_MEAN
FROM previous_application WHERE DAYS_DECISION <= 0 GROUP BY SK_ID_CURR

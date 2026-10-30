-- DELAY > 0 = paid after due date. Rows with payment/due date after application time are excluded (not knowable at assessment).
WITH i AS (
  SELECT SK_ID_CURR, AMT_INSTALMENT, AMT_PAYMENT, DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT AS DELAY
  FROM installments_payments
  WHERE DAYS_INSTALMENT <= 0 AND COALESCE(DAYS_ENTRY_PAYMENT, 0) <= 0
)
SELECT SK_ID_CURR,
  COUNT(*)                                                         AS INST_RECORD_COUNT,
  AVG(DELAY)                                                       AS AVERAGE_PAYMENT_DELAY,
  MAX(DELAY)                                                       AS MAX_PAYMENT_DELAY,
  AVG(CASE WHEN DELAY IS NULL THEN NULL WHEN DELAY > 0 THEN 1.0 ELSE 0.0 END) AS LATE_PAYMENT_RATIO,
  SUM(AMT_PAYMENT) / NULLIF(SUM(AMT_INSTALMENT), 0)                AS INSTALLMENT_PAYMENT_RATIO,
  AVG(CASE WHEN AMT_PAYMENT < AMT_INSTALMENT THEN 1.0 ELSE 0.0 END) AS INST_UNDERPAID_RATIO
FROM i GROUP BY SK_ID_CURR

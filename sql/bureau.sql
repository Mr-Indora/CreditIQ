-- Bureau + bureau_balance -> one row per SK_ID_CURR.
-- Leakage guard: DAYS_ENDDATE_FACT > 0 means "closed after application" -> neutralised; rows with DAYS_CREDIT > 0 dropped.
WITH bb AS (
  SELECT SK_ID_BUREAU,
         COUNT(*) AS BB_MONTHS,
         AVG(CASE WHEN STATUS IN ('1','2','3','4','5') THEN 1.0 ELSE 0.0 END) AS BB_DPD_RATIO,
         MAX(CASE WHEN STATUS IN ('1','2','3','4','5') THEN CAST(STATUS AS INTEGER) ELSE 0 END) AS BB_MAX_DPD_STATUS
  FROM bureau_balance WHERE MONTHS_BALANCE <= 0 GROUP BY SK_ID_BUREAU
), b AS (
  SELECT br.SK_ID_CURR, br.SK_ID_BUREAU, br.CREDIT_ACTIVE, br.DAYS_CREDIT, br.CREDIT_DAY_OVERDUE,
         br.AMT_CREDIT_MAX_OVERDUE, br.CNT_CREDIT_PROLONG, br.AMT_CREDIT_SUM, br.AMT_CREDIT_SUM_DEBT,
         br.AMT_CREDIT_SUM_OVERDUE, bb.BB_DPD_RATIO, bb.BB_MAX_DPD_STATUS,
         CASE WHEN br.DAYS_ENDDATE_FACT > 0 THEN NULL ELSE br.DAYS_ENDDATE_FACT END AS DAYS_ENDDATE_FACT_SAFE
  FROM bureau br LEFT JOIN bb ON br.SK_ID_BUREAU = bb.SK_ID_BUREAU
  WHERE br.DAYS_CREDIT <= 0
)
SELECT SK_ID_CURR,
  COUNT(*)                                                         AS BUREAU_PREV_CREDIT_COUNT,
  SUM(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 ELSE 0 END)        AS BUREAU_ACTIVE_CREDIT_COUNT,
  SUM(CASE WHEN CREDIT_ACTIVE = 'Closed' THEN 1 ELSE 0 END)        AS BUREAU_CLOSED_CREDIT_COUNT,
  AVG(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1.0 ELSE 0.0 END)    AS ACTIVE_CREDIT_RATIO,
  SUM(AMT_CREDIT_SUM)                                              AS BUREAU_TOTAL_CREDIT_EXPOSURE,
  AVG(AMT_CREDIT_SUM_DEBT)                                         AS AVERAGE_CREDIT_DEBT,
  SUM(AMT_CREDIT_SUM_DEBT)                                         AS BUREAU_TOTAL_DEBT,
  SUM(AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(AMT_CREDIT_SUM), 0)        AS BUREAU_DEBT_CREDIT_RATIO,
  AVG(CASE WHEN COALESCE(CREDIT_DAY_OVERDUE,0) > 0 OR COALESCE(AMT_CREDIT_SUM_OVERDUE,0) > 0
           THEN 1.0 ELSE 0.0 END)                                  AS OVERDUE_ACCOUNT_RATIO,
  MAX(CREDIT_DAY_OVERDUE)                                          AS BUREAU_MAX_DAYS_OVERDUE,
  MAX(AMT_CREDIT_MAX_OVERDUE)                                      AS BUREAU_MAX_OVERDUE_AMT,
  SUM(CNT_CREDIT_PROLONG)                                          AS BUREAU_PROLONG_COUNT,
  AVG(DAYS_CREDIT)                                                 AS BUREAU_DAYS_CREDIT_MEAN,
  MIN(DAYS_CREDIT)                                                 AS BUREAU_DAYS_CREDIT_MIN,
  AVG(DAYS_ENDDATE_FACT_SAFE)                                      AS BUREAU_DAYS_ENDDATE_FACT_MEAN,
  AVG(BB_DPD_RATIO)                                                AS BUREAU_BB_DPD_RATIO_MEAN,
  MAX(BB_MAX_DPD_STATUS)                                           AS BUREAU_BB_MAX_DPD_STATUS
FROM b GROUP BY SK_ID_CURR

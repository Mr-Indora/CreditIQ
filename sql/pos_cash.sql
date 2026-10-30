SELECT SK_ID_CURR,
  COUNT(*)                                                       AS POS_RECORD_COUNT,
  COUNT(DISTINCT SK_ID_PREV)                                     AS POS_LOAN_COUNT,
  AVG(SK_DPD)                                                    AS POS_DPD_MEAN,
  MAX(SK_DPD)                                                    AS POS_DPD_MAX,
  MAX(SK_DPD_DEF)                                                AS POS_DPD_DEF_MAX,
  AVG(CASE WHEN SK_DPD > 0 THEN 1.0 ELSE 0.0 END)                AS POS_LATE_RATIO,
  AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Active' THEN 1.0 ELSE 0.0 END) AS POS_ACTIVE_RATIO
FROM POS_CASH_balance WHERE MONTHS_BALANCE <= 0 GROUP BY SK_ID_CURR

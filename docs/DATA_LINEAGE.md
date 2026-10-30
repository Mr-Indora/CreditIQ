# Data Lineage
| Output | Built from | Key / grain |
|---|---|---|
| agg_bureau | bureau + bureau_balance (DAYS_CREDIT<=0; DAYS_ENDDATE_FACT>0 nulled) | SK_ID_CURR |
| agg_previous_application | previous_application (DAYS_DECISION<=0) | SK_ID_CURR |
| agg_pos_cash | POS_CASH_balance (MONTHS_BALANCE<=0) | SK_ID_CURR |
| agg_installments | installments_payments (due/payment offsets <=0) | SK_ID_CURR |
| agg_credit_card | credit_card_balance (MONTHS_BALANCE<=0) | SK_ID_CURR |
| features_train / features_test | application_{train,test} LEFT JOIN each agg (validate one_to_one) + ratios + HAS_* flags | SK_ID_CURR |
| holdout_features / holdout_scored | untouched 20% of features_train + model PD | SK_ID_CURR |

Count columns (`*_COUNT`) are 0 when an applicant has no history; ratios stay NaN (and `HAS_*` flags carry the information).

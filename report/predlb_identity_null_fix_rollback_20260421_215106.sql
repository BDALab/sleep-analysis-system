-- Rollback script for null-fix diagnosis_code updates from preDLB_shared.xlsx Identity sheet
BEGIN
TRANSACTION;
UPDATE dashboard_subject
SET diagnosis_code = NULL
WHERE id = 190;
UPDATE dashboard_subject
SET diagnosis_code = NULL
WHERE id = 193;
UPDATE dashboard_subject
SET diagnosis_code = NULL
WHERE id = 198;
UPDATE dashboard_subject
SET diagnosis_code = NULL
WHERE id = 200;
UPDATE dashboard_subject
SET diagnosis_code = NULL
WHERE id = 204;
UPDATE dashboard_subject
SET diagnosis_code = NULL
WHERE id = 232;
UPDATE dashboard_subject
SET diagnosis_code = NULL
WHERE id = 237;
UPDATE dashboard_subject
SET diagnosis_code = NULL
WHERE id = 241;
UPDATE dashboard_subject
SET diagnosis_code = NULL
WHERE id = 243;
COMMIT;

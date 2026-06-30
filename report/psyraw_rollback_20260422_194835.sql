-- Rollback script for PSY_RAW baseline diagnosis_code update
BEGIN
TRANSACTION;
UPDATE dashboard_subject
SET diagnosis_code = 1
WHERE id = 290;
UPDATE dashboard_subject
SET diagnosis_code = 0
WHERE id = 197;
UPDATE dashboard_subject
SET diagnosis_code = 1
WHERE id = 153;
UPDATE dashboard_subject
SET diagnosis_code = 0
WHERE id = 156;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 189;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 48;
UPDATE dashboard_subject
SET diagnosis_code = 2
WHERE id = 175;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 270;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 271;
UPDATE dashboard_subject
SET diagnosis_code = 1
WHERE id = 59;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 61;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 62;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 64;
UPDATE dashboard_subject
SET diagnosis_code = 1
WHERE id = 67;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 68;
UPDATE dashboard_subject
SET diagnosis_code = 1
WHERE id = 70;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 73;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 79;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 80;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 104;
UPDATE dashboard_subject
SET diagnosis_code = 1
WHERE id = 105;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 84;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 106;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 88;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 90;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 98;
UPDATE dashboard_subject
SET diagnosis_code = 0
WHERE id = 150;
UPDATE dashboard_subject
SET diagnosis_code = 1
WHERE id = 127;
UPDATE dashboard_subject
SET diagnosis_code = 2
WHERE id = 128;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 133;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 136;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 55;
UPDATE dashboard_subject
SET diagnosis_code = 0
WHERE id = 145;
UPDATE dashboard_subject
SET diagnosis_code = 0
WHERE id = 56;
UPDATE dashboard_subject
SET diagnosis_code = 1
WHERE id = 146;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 211;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 231;
UPDATE dashboard_subject
SET diagnosis_code = 1
WHERE id = 233;
UPDATE dashboard_subject
SET diagnosis_code = 1
WHERE id = 235;
UPDATE dashboard_subject
SET diagnosis_code = 1
WHERE id = 239;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 240;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 241;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 242;
UPDATE dashboard_subject
SET diagnosis_code = 0
WHERE id = 276;
UPDATE dashboard_subject
SET diagnosis_code = 1
WHERE id = 278;
UPDATE dashboard_subject
SET diagnosis_code = 2
WHERE id = 279;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 282;
UPDATE dashboard_subject
SET diagnosis_code = 3
WHERE id = 230;
COMMIT;

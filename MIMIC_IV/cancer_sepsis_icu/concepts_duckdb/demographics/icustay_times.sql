-- mimiciv_derived.icustay_times

DROP TABLE IF EXISTS mimiciv_derived.icustay_times;
CREATE TABLE mimiciv_derived.icustay_times AS

WITH t1 AS (
    SELECT
        stay_id,
        MIN(charttime) AS intime_hr,
        MAX(charttime) AS outtime_hr
    FROM mimiciv_icu.chartevents
    WHERE itemid = 220045
    GROUP BY stay_id
)

SELECT
    ie.subject_id,
    ie.hadm_id,
    ie.stay_id,
    t1.intime_hr,
    t1.outtime_hr
FROM mimiciv_icu.icustays ie
LEFT JOIN t1
    ON ie.stay_id = t1.stay_id;
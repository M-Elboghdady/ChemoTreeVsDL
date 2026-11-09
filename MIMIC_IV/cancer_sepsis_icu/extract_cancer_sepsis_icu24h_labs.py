from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from config.constants import MIMIC_DIR

M=Path(MIMIC_DIR)

COHORT=ROOT/"MIMIC_IV/saved_data/cohorts/mimic_cohort_cancer_sepsis_icu24h.csv.gz"
OUT=ROOT/"MIMIC_IV/saved_data/processed_admission_features_for_ts/mimic_cohort_cancer_sepsis_icu24h/mimic_cohort_cancer_sepsis_icu24h_admissions_labs_1_days_to_ts.csv.gz"

cohort=pd.read_csv(
    COHORT,
    usecols=["subject_id","hadm_id","source_hadm_id","stay_id","intime"],
    parse_dates=["intime"]
)

# One selected ICU stay per hospital admission.
assert cohort.source_hadm_id.is_unique

mapping=cohort.rename(columns={"subject_id":"cohort_subject_id"})

OUT.parent.mkdir(parents=True,exist_ok=True)
if OUT.exists(): OUT.unlink()

first=True
n=0

for chunk in pd.read_csv(
    M/"hosp/labevents.csv.gz",
    usecols=["subject_id","hadm_id","itemid","charttime","valuenum"],
    parse_dates=["charttime"],
    chunksize=1_000_000
):
    chunk=chunk[
        chunk.hadm_id.isin(mapping.source_hadm_id) &
        chunk.valuenum.notna()
    ].copy()

    if chunk.empty:
        continue

    chunk=chunk.merge(
        mapping,
        left_on="hadm_id",
        right_on="source_hadm_id",
        how="inner"
    )

    # Safety: same patient.
    chunk=chunk[chunk.subject_id==chunk.cohort_subject_id].copy()

    chunk["minute"]=(
        (chunk.charttime-chunk.intime).dt.total_seconds()//60
    ).astype(int)

    # First 24 h after ICU admission.
    chunk=chunk[(chunk.minute>=0)&(chunk.minute<1440)].copy()

    if chunk.empty:
        continue

    chunk["value"]=chunk.valuenum.astype(float)
    chunk["hour"]=chunk.minute//60
    chunk["day"]=chunk.minute//1440

    out=chunk[[
        "cohort_subject_id",
        "hadm_id_y",
        "source_hadm_id",
        "stay_id",
        "itemid",
        "value",
        "minute",
        "hour",
        "day"
    ]].rename(columns={
        "cohort_subject_id":"subject_id",
        "hadm_id_y":"hadm_id"
    })

    out.to_csv(
        OUT,
        mode="wt" if first else "at",
        header=first,
        index=False,
        compression="gzip"
    )

    first=False
    n+=len(out)
    print("Events written:",n)

print("\nSaved:",OUT)

df=pd.read_csv(OUT)
print("Events:",len(df))
print("Samples with labs:",df.hadm_id.nunique())
print("Patients with labs:",df.subject_id.nunique())
print("Unique labs:",df.itemid.nunique())
print("Minute range:",df.minute.min(),df.minute.max())

import sys,re
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from config.constants import MIMIC_DIR

M=Path(MIMIC_DIR)
SEPSIS=ROOT/"MIMIC_IV/cancer_sepsis_icu/sepsis3.csv.gz"
OUT=ROOT/"MIMIC_IV/saved_data/cohorts/mimic_cohort_cancer_sepsis_icu24h.csv.gz"

def is_cancer(code,v):
    if pd.isna(code) or pd.isna(v): return False
    c=str(code).replace(".","").strip().upper()
    try: v=int(v)
    except: return False
    if v==9:
        x=c[:3]
        return x.isdigit() and 140<=int(x)<=209
    return v==10 and bool(re.match(r"^C(?:0\d|[1-8]\d|9[0-7])",c))

# ICU stays.
icu=pd.read_csv(
    M/"icu/icustays.csv.gz",
    usecols=["subject_id","hadm_id","stay_id","intime","outtime"],
    parse_dates=["intime","outtime"]
)

# Paper: only FIRST ICU admission within each hospital stay.
icu=(
    icu.sort_values(["hadm_id","intime","stay_id"])
       .drop_duplicates("hadm_id",keep="first")
       .copy()
)

# ICU duration >24 h.
icu=icu[(icu.outtime-icu.intime)>pd.Timedelta(hours=24)].copy()

# Adults: paper flowchart excludes age <18.
pat=pd.read_csv(
    M/"hosp/patients.csv.gz",
    usecols=["subject_id","gender","anchor_age","anchor_year"]
)
icu=icu.merge(pat,on="subject_id",how="inner")
icu["age"]=icu.anchor_age+icu.intime.dt.year-icu.anchor_year
icu=icu[icu.age>=18].copy()

# Cancer ICD-9 140-209 / ICD-10 C00-C97.
dx=pd.read_csv(
    M/"hosp/diagnoses_icd.csv.gz",
    usecols=["hadm_id","icd_code","icd_version"],
    dtype={"icd_code":"string"}
)
dx["cancer"]=[is_cancer(c,v) for c,v in zip(dx.icd_code,dx.icd_version)]
icu=icu[icu.hadm_id.isin(set(dx.loc[dx.cancer,"hadm_id"]))].copy()

# Sepsis-3.
s3=pd.read_csv(SEPSIS)
if s3["sepsis3"].dtype==bool:
    s3=s3[s3.sepsis3]
else:
    s3=s3[s3.sepsis3.astype(str).str.lower().isin(["1","true"])]

icu=icu.merge(
    s3[["stay_id"]].drop_duplicates(),
    on="stay_id",how="inner"
)

# Hospital mortality.
adm=pd.read_csv(
    M/"hosp/admissions.csv.gz",
    usecols=["subject_id","hadm_id","admittime","dischtime","hospital_expire_flag"],
    parse_dates=["admittime","dischtime"]
)
icu=icu.merge(adm,on=["subject_id","hadm_id"],how="inner")
icu["label"]=icu.hospital_expire_flag.astype(int)

# Pipeline sample ID = selected first ICU stay.
icu=icu.rename(columns={"hadm_id":"source_hadm_id"})
icu["hadm_id"]=icu.stay_id

cohort=icu[[
    "subject_id","hadm_id","source_hadm_id","stay_id",
    "admittime","dischtime","intime","outtime",
    "age","gender","label"
]].reset_index(drop=True)

assert cohort.stay_id.is_unique
assert cohort.source_hadm_id.is_unique
assert cohort.hadm_id.is_unique
assert (cohort.hadm_id==cohort.stay_id).all()
assert cohort.label.isin([0,1]).all()

OUT.parent.mkdir(parents=True,exist_ok=True)
cohort.to_csv(OUT,index=False,compression="gzip")

print("Samples:",len(cohort))
print("Admissions:",cohort.source_hadm_id.nunique())
print("Patients:",cohort.subject_id.nunique())
print("Deaths:",int(cohort.label.sum()))
print("Mortality:",round(cohort.label.mean(),4))
print("Saved:",OUT)

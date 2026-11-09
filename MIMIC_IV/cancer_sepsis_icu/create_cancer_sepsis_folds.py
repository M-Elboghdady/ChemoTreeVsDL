from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

ROOT=Path(__file__).resolve().parents[2]
COHORT=ROOT/"MIMIC_IV/saved_data/cohorts/mimic_cohort_cancer_sepsis_icu24h.csv.gz"
OUT=ROOT/"MIMIC_IV/saved_data/folds/mimic_cohort_cancer_sepsis_icu24h"

df=pd.read_csv(COHORT,usecols=["subject_id","hadm_id","label"])
OUT.mkdir(parents=True,exist_ok=True)

outer=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)

for fold,(dev_idx,test_idx) in enumerate(
    outer.split(df,df.label,groups=df.subject_id)
):
    dev=df.iloc[dev_idx].reset_index(drop=True)
    test=df.iloc[test_idx].reset_index(drop=True)

    # ~80/20 split of development set -> overall ≈64/16/20
    gss=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=42+fold)
    train_idx,val_idx=next(
        gss.split(dev,dev.label,groups=dev.subject_id)
    )

    train=dev.iloc[train_idx]
    val=dev.iloc[val_idx]

    # strict patient leakage checks
    tr=set(train.subject_id); va=set(val.subject_id); te=set(test.subject_id)
    assert not tr&va and not tr&te and not va&te

    def ids(x):
        return x[["subject_id","hadm_id"]].to_numpy(dtype=np.int64)

    with open(OUT/f"fold_{fold}.pkl","wb") as f:
        pickle.dump([ids(train),ids(val),ids(test)],f)

    print(
        f"Fold {fold}: "
        f"train={len(train)} ({train.label.mean():.3f}) | "
        f"val={len(val)} ({val.label.mean():.3f}) | "
        f"test={len(test)} ({test.label.mean():.3f})"
    )

print("Saved:",OUT)

import pandas as pd
import numpy as np
import sqlite3
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from config.constants import PROJECT_ROOT, MIMIC_DIR, UKER_DIR

class Extractor:
    
    def __init__(self, args):
        
        self.args = args
        self.dataset = args.dataset
        self.days_before_discharge = args.days_before_discharge  
        self.cohort = args.cohort
        self.itemids_to_remove = [50934, 50947, 51678]  
        self.set_paths()
        
        self.args.logger.write(f"Extractor created for dataset={self.dataset}, cohort={self.cohort}, days={self.days_before_discharge}")
        
        if not self.data_path.exists():
            self.extract_labs()
        else:
            self.args.logger.write(f"Data already extracted, saved in {self.data_path}")

    def extract_labs(self):
        if self.dataset == "MIMIC_IV":
            self.extract_mimic_labs()
        elif self.dataset == "UKEr":
            self.extract_uker_labs() 
        else:
            raise ValueError("Extraction supported for MIMIC and UKER datasets only!")

    def extract_mimic_labs(self):
        raise NotImplementedError   
    
    def extract_uker_labs(self):
        raise NotImplementedError   

    def set_paths(self):
        # saved data path
        self.saved_data_path = PROJECT_ROOT / self.dataset / "saved_data"
        # define features name
        self.input_features = f"{self.cohort}_admissions_labs_{self.days_before_discharge}_days"
        # cohort path (with demo graph data and target for training)
        self.cohort_path = self.saved_data_path / "cohorts" / f"{self.cohort}.csv.gz"
        self.features_path = self.saved_data_path / "features" / f"{self.input_features}.csv.gz"
        # temporal data path
        self.data_dir = self.saved_data_path / "processed_admission_features_for_ts" / self.cohort 
        self.data_dir.mkdir(parents=True, exist_ok=True)    
        self.data_path = self.data_dir / f"{self.input_features}_to_ts.csv.gz"
        # CV split path
        self.CV_folds_dir = self.saved_data_path / "folds" / self.cohort
        self.CV_folds_dir.mkdir(parents=True, exist_ok=True)
        

class ExtractorTrain(Extractor):
    def __init__(self, args):
        super().__init__(args)
        
    def extract_mimic_labs(self):
        if self.cohort=="mimic_cohort_cancer_sepsis_icu24h":
            return self.extract_mimic_icu24h_labs()

        admission_labs = pd.read_csv(
            self.features_path,
            compression="gzip",
            header=0,
            index_col=None,
            parse_dates=["dischtime", "date"]
        )
        admission_labs = admission_labs[~admission_labs["itemid"].isin(self.itemids_to_remove)]
        
        admission_labs["starttime"] = (admission_labs["dischtime"] - pd.DateOffset(days=self.days_before_discharge)).apply(
            lambda x: x.replace(hour=0, minute=0, second=0)
        )

        admission_labs["minute"] = (admission_labs["date"] - admission_labs["starttime"]).dt.total_seconds() / 60
        admission_labs["hour"] = admission_labs["minute"] / 60
        admission_labs["day"] = admission_labs["hour"] / 24
        admission_labs[["minute", "hour", "day"]] = admission_labs[["minute", "hour", "day"]].astype("int")

        # save to file
        admission_labs[["subject_id", "hadm_id", "itemid", "value", "minute", "hour", "day"]].to_csv(
            self.data_path, index=False, compression="gzip"
        )
        self.args.logger.write(f"Labs processed and saved to {self.data_path}")
    def extract_mimic_icu24h_labs(self):
        c=pd.read_csv(self.cohort_path,compression="gzip",
            usecols=["subject_id","hadm_id","source_hadm_id","intime"],
            parse_dates=["intime"])
        ref=c[["subject_id","source_hadm_id","hadm_id","intime"]]
        hadms=set(c.source_hadm_id)
        raw=Path(MIMIC_DIR)/"hosp/labevents.csv.gz"

        if self.data_path.exists(): self.data_path.unlink()
        first=True;n=0

        for x in pd.read_csv(raw,
            usecols=["subject_id","hadm_id","itemid","charttime","valuenum"],
            chunksize=1_000_000):

            x=x[x.hadm_id.isin(hadms)&x.valuenum.notna()&x.charttime.notna()&
                ~x.itemid.isin(self.itemids_to_remove)].copy()
            if x.empty: continue

            x["charttime"]=pd.to_datetime(x.charttime,format="mixed")
            x=x.rename(columns={"hadm_id":"source_hadm_id"})
            x=x.merge(ref,on=["subject_id","source_hadm_id"])

            x["minute"]=(x.charttime-x.intime).dt.total_seconds()/60
            x=x[(x.minute>=0)&(x.minute<1440)].copy()
            if x.empty: continue

            x["minute"]=x.minute.astype(int)
            x["hour"]=(x.minute//60).astype(int)
            x["day"]=0
            x=x.rename(columns={"valuenum":"value"})

            y=x[["subject_id","hadm_id","itemid","value","minute","hour","day"]]
            y.to_csv(self.data_path,mode="a",header=first,index=False,compression="gzip")
            first=False;n+=len(y)

        if first: raise RuntimeError("No first-24h ICU labs found.")
        self.args.logger.write(f"ICU first-24h labs: {n} events saved to {self.data_path}")

    def extract_uker_labs(self):
        admission_labs = pd.read_csv(
            self.features_path,
            compression="gzip",
            header=0,
            index_col=None
        )
        
        admission_labs["starttime"] = admission_labs["dischtime"] - self.days_before_discharge
    
        admission_labs["day"] = admission_labs["date"] - admission_labs["starttime"]
        admission_labs["hour"] = admission_labs["day"]*24
        admission_labs["minute"] = admission_labs["day"]*24*60
        admission_labs[["minute", "hour", "day"]] = admission_labs[["minute", "hour", "day"]].astype("int")
    
        # save to file
        admission_labs[["subject_id", "hadm_id", "itemid", "value", "minute", "hour", "day"]].to_csv(
            self.data_path, index=False, compression="gzip"
        )
        self.args.logger.write(f"Labs processed and saved to {self.data_path}")
            
            
class ExtractorPretrain(Extractor): 
    # class to extract admission data for pretraining tasks
    def __init__(self, args):
        self.mimic_path, self.uker_path = Path(MIMIC_DIR), Path(UKER_DIR)
        super().__init__(args)

    def get_uker_all_cohort(self):
        query = f""" 
        SELECT f.*, p.sex
        FROM fall f
        JOIN pat p ON f.pid = p.pid;
        """
        adms = self._read_SQL(UKER_DIR, query)
        adms["age"] = adms['aufnahme_alter']
        adms.columns =  ['hadm_id','subject_id','admittime' ,'dischtime', 'gender', 'age']
        adms.to_csv(self.cohort_path, compression="gzip", index=False)
        self.args.logger.write("Admission and demographic data preprocessed.")

    def get_mimic_all_cohort(self):
        # read admissions and discharge times
        adms = pd.read_csv(
            self.mimic_path / "hosp/admissions.csv.gz",
            compression="gzip", 
            usecols=["subject_id", "hadm_id", "dischtime"], 
            dtype={"subject_id": "int64", "hadm_id": "int64"}, 
            parse_dates=["dischtime"]
        ).drop_duplicates()
        self.args.logger.write("Admission data loaded.")
        
        demo = pd.read_csv(
            self.mimic_path / "hosp/patients.csv.gz",
            compression="gzip", 
            usecols=["subject_id", "gender", "anchor_age"]
        ).rename(columns={"anchor_age": "age"})
        
        demo = demo.merge(adms, on="subject_id", how="inner")
        demo.to_csv(self.cohort_path, compression="gzip", index=False)
        self.args.logger.write("Demographic data preprocessed.")
    
    def extract_uker_labs(self):
        # if not cohort is specified, set cohort to whole MIMIC
        if self.cohort == "uker_all" and not self.cohort_path.exists():
            self.get_uker_all_cohort()    
            
        adms = pd.read_csv(
            self.cohort_path,
            compression='gzip',
            usecols=["subject_id", "hadm_id", "dischtime"]
        )
    
        adms["starttime"] = adms["dischtime"] - self.days_before_discharge
    
        # 12 chunks
        first = True
        pt_ids = []
        
        conn = sqlite3.connect(self.uker_path)
 
        chunksize = 1000000
        for chunk in pd.read_sql("SELECT * FROM lab", conn, chunksize=chunksize):
            chunk.columns = ["subject_id", "itemid", "date", "value"]
            # retain labs within time window
            sub_labs = chunk.merge(adms, on="subject_id")
            sub_labs = sub_labs[(sub_labs["date"]>=sub_labs["starttime"]) & (sub_labs["date"]<=sub_labs["dischtime"])]
            # compute minutes
            sub_labs["day"] = sub_labs["date"] - sub_labs["starttime"]
            sub_labs["hour"] = sub_labs["day"]*24
            sub_labs["minute"] = sub_labs["day"]*24*60
            sub_labs[["minute", "hour", "day"]] = sub_labs[["minute", "hour", "day"]].astype("int")
            
            pt_ids.append(sub_labs[["subject_id","hadm_id"]].drop_duplicates())
            
            sub_labs[["subject_id", "hadm_id", "itemid", "value", "minute", "day"]].to_csv(
                self.data_path,
                mode="a",
                header=first,
                index=False,
                compression="gzip"
            )
            first = False
        
        #self.args.logger.write("Lab data loaded and preprocessed.")
        
        # get ids with labs
        pt_ids = pd.concat(pt_ids, ignore_index=True).drop_duplicates()
        
        # get cohort ids
        cohort_subj_ids = np.concatenate([
            pd.read_csv(self.saved_data_path / 'cohorts' / 'uker_cohort_NF_30_days.csv.gz', compression='gzip')["subject_id"].values,
            pd.read_csv(self.saved_data_path / 'cohorts' / 'uker_cohort_aplasia_45_days.csv.gz', compression='gzip')["subject_id"].values
        ])
        
        # remove ids in training cohorts
        pt_ids = pt_ids[~pt_ids["subject_id"].isin(cohort_subj_ids)].reset_index(drop=True)
        pt_ids.to_csv(self.CV_folds_dir / "pt_ids.csv")
        
        # apply split
        self._get_ids_dict(pt_ids, 0.80)
        self.args.logger.write("CV splits set.")

    def recompute_split(self, tsize):
        import pandas as pd
    
        try:
            pt_ids = pd.read_csv(self.CV_folds_dir / "pt_ids.csv", usecols=["subject_id","hadm_id"])
        except FileNotFoundError:
            raise RuntimeError("cohort not extracted! Missing pt_ids.csv")
    
        self._get_ids_dict(pt_ids, tsize)
        self.args.logger.write("CV splits set.")
        
    def extract_mimic_labs(self):
        
        # if not cohort is specified, set cohort to whole MIMIC
        if self.cohort == "mimic_all" and not self.cohort_path.exists():
            self.args.logger.write("Extracting uker cohort data.")
            self.get_mimic_all_cohort()
                
        adms = pd.read_csv(
            self.cohort_path,
            compression='gzip',
            parse_dates=['dischtime'],
            usecols=["subject_id", "hadm_id", "dischtime"]
        )
        # extract all labs within X days to discharge time
        adms["time_before_disch"] = adms["dischtime"] - pd.DateOffset(days=self.days_before_discharge)
        adms["starttime"] = adms["time_before_disch"].apply(lambda x: x.replace(hour=0, minute=0, second=0))
        
        # 159 chunks
        first = True
        pt_ids = []

        for chunk in tqdm(pd.read_csv(
                self.mimic_path / "hosp/labevents.csv.gz", compression='gzip', 
                usecols=['subject_id','itemid','valuenum','valueuom','charttime'], 
                dtype={'subject_id':'int64', 'itemid':'int64', 'valuenum':'float64', 'valueuom':'object'}, 
                parse_dates=['charttime'],
                chunksize=1000000)):
            
            # remove items with null values
            chunk = chunk.dropna(subset=['valuenum'])
            # remove items not clinically relevant
            chunk = chunk[~chunk['itemid'].isin(self.itemids_to_remove)]
            # select labs before discharge
            sub_labs = chunk.merge(adms, on="subject_id")
            sub_labs = sub_labs[(sub_labs["charttime"]>=sub_labs["time_before_disch"]) & (sub_labs["charttime"]<=sub_labs["dischtime"])]
            # compute minutes from start time
            sub_labs["minute"] = (sub_labs["charttime"] - sub_labs["starttime"]).dt.total_seconds() / 60
            sub_labs = sub_labs.rename(columns={"valuenum":"value"})
            
            pt_ids.append(sub_labs[["subject_id","hadm_id"]].drop_duplicates())
            
            sub_labs[["subject_id", "hadm_id", "itemid", "value", "minute"]].to_csv(
                self.data_path,
                mode="a",
                header=first,
                index=False,
                compression="gzip"
            )
            first = False
        self.args.logger.write("Lab data loaded and preprocessed.")
        
        # get ids with labs
        pt_ids = pd.concat(pt_ids, ignore_index=True).drop_duplicates()

        # get cohort ids
        cohort_subj_ids = np.concatenate([
            pd.read_csv(self.saved_data_path / 'cohorts' / 'mimic_cohort_NF_30_days.csv.gz', compression='gzip')["subject_id"].values,
            pd.read_csv(self.saved_data_path / 'cohorts' / 'mimic_cohort_aplasia_45_days.csv.gz', compression='gzip')["subject_id"].values,
            pd.read_csv(self.saved_data_path / 'cohorts' / 'mimic_cohort_cancer_sepsis_icu24h.csv.gz', compression='gzip')["subject_id"].values
        ])
        cohort_subj_ids = np.unique(cohort_subj_ids)
        # remove ids in training cohorts
        pt_ids = pt_ids[~pt_ids["subject_id"].isin(cohort_subj_ids)].reset_index(drop=True)
        pt_ids.to_csv(self.CV_folds_dir / "pt_ids.csv")
        
        # apply split
        self._get_ids_dict(pt_ids, 0.80)
        self.args.logger.write("CV splits set.")       

    def _read_SQL(self, data_path, query):
        with sqlite3.connect(data_path) as conn:
            df = pd.read_sql(query, conn)
        return df
        
    def _get_ids_dict(self, pt_ids, tsize):
        
        groups = pt_ids["subject_id"].values
        admids = pt_ids["hadm_id"].values

        gss = GroupShuffleSplit(n_splits=1, train_size=tsize, test_size=1-tsize, random_state=42)

        train_idx, val_idx = next(gss.split(pt_ids, groups=groups))
        test_idx = np.array([], dtype=int)

        train_ids = admids[train_idx]
        val_ids = admids[val_idx]
        test_ids = np.array([], dtype=int)
        
        print(f"train size = {len(train_ids)}")
        print(f"val size   = {len(val_ids)}")
        print(f"test size  = {len(test_ids)}")

        pt_dict_ids = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids
        }

        with open(self.CV_folds_dir / "pt_dict_ids.pkl", "wb") as f:
            pickle.dump(pt_dict_ids, f)
            
        train_hadms = pt_ids.iloc[train_idx].values
        val_hadms = pt_ids.iloc[val_idx].values
        test_hadms = pt_ids.iloc[test_idx].values
        
        # single split for pretraining
        with open(self.CV_folds_dir / "fold_0.pkl", "wb") as f:   
            pickle.dump([train_hadms, val_hadms, test_hadms], f)






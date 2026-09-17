import pandas as pd
import numpy as np
import pickle
from ts_model_training.utils import safe_pos_freq, remove_features_not_in_train, compute_lab_frequency, ids_in_data, set_splits, compute_class_weight
from ts_model_training.preprocessor import PreprocessorA, PreprocessorB, PreprocessorC_sup, PreprocessorC_unsup, PreprocessorEHRMamba

from config.constants import PROJECT_ROOT

class TimeSeriesDataset:
    def __init__(self, args):
        self.args = args
        self.load_data()
        self.get_temporal_data()
        self.get_static_data()
        self.get_target()    
        self.preprocess_data()

    def load_data(self):
        # load target cohort
        self.cohort = pd.read_csv(self.args.paths["cohort_path"], compression='gzip')

        # load time series data with dtype specifications
        dtype_spec = {
            'subject_id': 'int64',
            'hadm_id': 'int64',
            'minute': 'int64',
            'hour': 'int64',
            'day': 'int64',
            'itemid': 'string',
            'value': 'float64'
        }
        # load data
        self.data = pd.read_csv(self.args.paths["data_path"], dtype=dtype_spec)#, nrows=2_000_000)
        # do not consider minutes but only days
        if getattr(self.args, "drop_minutes", False):
            self.data["minute"] = self.data["day"]*24*60
        # keep ids found in data
        self.data, self.args.ids = ids_in_data(self.data, self.args.ids, self.args.logger)
        self.args.N = len(self.args.ids["sup_ids"])
        # save splits
        self.ts_id_to_ind, self.splits = set_splits(self.args.ids)
        self.data['ts_ind'] = self.data['hadm_id'].map(self.ts_id_to_ind)


    def get_temporal_data(self):
        
        # select features specified by file
        if self.args.feature_threshold:
            #feature_file = self.args.paths["feature_path"] / f'fold_{self.args.fold}_features.pkl'
            feature_file = self.args.paths["feature_path"] / 'mimic_top100_features.pkl'
            with open(feature_file, 'rb') as f:
                self.selected_features = list(map(str, pickle.load(f)))
            self.data = self.data[self.data["itemid"].isin(self.selected_features)]
            self.args.logger.write('\nFeature selected from file: ' + str(len(self.selected_features)))
            
        # remove features not present in training data
        self.data = remove_features_not_in_train(self.data, self.args.ids["train"], self.args.logger)

        # select features specified by parameter
        self.static_labs = None
        if self.args.static_threshold > 0:
            data = self.data
            ts_items = compute_lab_frequency(data[data["hadm_id"].isin(self.args.ids["train"])])
            ts_items["type"] = np.where(ts_items["num_ts"] > self.args.static_threshold, "temporal", "static")
            data = data.merge(ts_items[["itemid","type"]], how ="right")
            self.static_labs = (data[data["type"] == "static"].groupby(["hadm_id", "itemid"], as_index=False)["value"].mean())
            # TO DO - add demographic
            self.data = data[data["type"] == "temporal"].drop(columns=["type"])
            self.args.logger.write('\nFeatures split based on min frequency: '+ str(self.args.static_threshold))
            self.args.logger.write('Temporal features: '+str(self.data.itemid.nunique()))
            self.args.logger.write('Static features: '+str(self.static_labs.itemid.nunique()))


    def get_static_data(self):
        demo_varis = ["age","gender"] # others can be added later
        static_data = self.cohort[["hadm_id"] + demo_varis].copy()
        static_data["gender"] = static_data["gender"].str.upper().map({"M":0,"F":1})
        static_data = static_data.melt(id_vars="hadm_id", var_name="itemid", value_name="value")
        static_data = static_data.loc[static_data.hadm_id.isin(self.args.ids["sup_ids"])]
        
        # add static labs
        if self.static_labs is not None:
            static_data = pd.concat([static_data, self.static_labs], ignore_index=True)
        static_data['ts_ind'] = static_data['hadm_id'].map(self.ts_id_to_ind)
        
        # reformat data
        self.static_data = static_data.pivot(index="ts_ind",columns="itemid",values="value")
        #self.args.D = len(self.static_data.columns)
        
        # standardise data
        demo_raw = self.static_data.values
        train_ind = self.splits["train"]
        demo_means = demo_raw[train_ind].mean(axis=0, keepdims=True)
        demo_stds = demo_raw[train_ind].std(axis=0, keepdims=True)
        demo_stds = np.where(demo_stds == 0, 1, demo_stds)
        demo_norm = (demo_raw - demo_means) / demo_stds
        demo_norm = np.nan_to_num(demo_norm, nan=0.0)
        
        # handle missing indicators
        cols_with_na = self.static_data.columns[self.static_data.isna().any()]
        if len(cols_with_na) > 0:
            missing_indicators = self.static_data[cols_with_na].notna().astype(int).values
            demo_norm = np.concatenate([demo_norm, missing_indicators], axis=1)       
        
        self.demo_means, self.demo_stds = demo_means, demo_stds
        self.demo_raw, self.demo = demo_raw, demo_norm
        self.args.D = self.demo.shape[1]
        
        self.args.logger.write('\nStatic variables: ' + ', '.join(self.static_data.columns))
        self.args.logger.write('Total static features: '+str(self.args.D))


    def get_target(self):
        if self.args.train_mode != "pretrain":
            # only compute target in supervised settings
            tg = self.cohort[['hadm_id', 'label']]
            tg = tg.loc[tg.hadm_id.isin(self.args.ids["sup_ids"])]
            tg['ts_ind'] = tg['hadm_id'].map(self.ts_id_to_ind)
            tg.sort_values('ts_ind', inplace=True)

            # save target     
            self.y = np.array(tg['label'])

            # compute class weight
            train_pos_freq = safe_pos_freq(self.y[self.splits['train']])
            val_pos_freq   = safe_pos_freq(self.y[self.splits['val']])
            test_pos_freq  = safe_pos_freq(self.y[self.splits['test']])

            self.args.pos_class_weight = compute_class_weight(train_pos_freq, self.args.pos_class_weight, self.args.stratify_batch, self.args.train_batch_size)

            self.args.logger.write('\npos class weight: ' + str(round(self.args.pos_class_weight, 2)))
            self.args.logger.write('% pos class in train, val, test splits: ' + 
                                str([round(x, 3) for x in [train_pos_freq, val_pos_freq, test_pos_freq]]))


    def preprocess_data(self):
        # get processor based on model
        model_type = self.args.model_type
        train_ind = self.splits['train']
        if model_type in ['gru', 'lstm', 'tcn', 'sand', 'mlp', 'mamba', 'mamba_fusion']:
            self.preproc = PreprocessorA(self)
	elif model_type == 'ehrmamba_lab':
    	    self.preproc = PreprocessorEHRMamba(self)
        elif model_type in ['grud', 'interpnet']:
            self.preproc = PreprocessorB(self)
        elif model_type in ['strats', 'istrats'] and self.args.train_mode == "pretrain":
            self.preproc = PreprocessorC_unsup(self)
        elif model_type in ['strats', 'istrats'] and self.args.train_mode != "pretrain":
            self.preproc = PreprocessorC_sup(self)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        # preprocess data accordingly
        self.preproc.prepare_inputs()
        # save input dict
        self.preproc.save_inputs()
        # remove problematic inputs


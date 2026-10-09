from itertools import product
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from math import ceil
import torch
import yaml
import os
from datetime import datetime
from ts_model_training.logger import Logger
from ts_model_training.utils import set_all_seeds, set_all_paths, load_fold_file
from config.constants import num_folds, num_inner_folds
from sklearn.model_selection import StratifiedGroupKFold

from ts_model_training.dataset import TimeSeriesDataset
from ts_model_training.trainer import Trainer
from ts_model_training.factory import build_model, build_batcher
from ts_model_training.extractor import ExtractorTrain, ExtractorPretrain
from ts_model_training.batcher import Batcher, BatcherA, BatcherB, BatcherC_sup, BatcherC_unsup

class EnvManager:
    def __init__(self, args):
        # set env config
        self.args = args
        self.nfolds = num_folds
        self.config_path = args.config_path or "ts_model_training/ts_config_params.yaml"

        # load config and initialise environment
        self.set_device()
        self.load_global_config()
        self.args.paths = set_all_paths(args)

        # added configuration based on loaded params
        self.set_stratify_batch()
        self.setup_logging()

        # set seeds
        set_all_seeds(args.seed, args.fast)


    def load_global_config(self):
        # load model parameters from YAML configuration.
        with open(self.config_path, "r") as f:
            model_params = yaml.safe_load(f)

        global_config = model_params.get("global")
        model_config = model_params.get(self.args.model_type)

        if not global_config:
            raise ValueError("Global config section 'global' not found in model_params.yaml")
        if not model_config:
            raise ValueError(f"Model config for '{self.args.model_type}' not found in model_params.yaml")

        self.param_grid = model_config.get("grid_search", {})
        self.default_params = model_config.get(self.args.cohort, {})
        self.default_params = self._check_param_list(self.default_params, ["hid_dims"])
        
        self._set_args_attributes(global_config)
        self.resolve_mode()

    def _check_param_list(self, param_dict, str_params):
        for sp in str_params:
            if sp in param_dict:
                s = param_dict[sp]
                if isinstance(s, str):
                    # safely strip brackets and split values
                    param_dict[sp] = [int(x) for x in s.strip("[]").split(",") if x.strip()]
        return param_dict 
        
    def _set_args_attributes(self, param_dict, log_message=None):
        # helper to set multiple attributes on self.args.
        for k, v in param_dict.items():
            setattr(self.args, k, v)
        if log_message and hasattr(self.args, 'logger'):
            self.args.logger.write(log_message)

    def set_model_params(self, mode="default", param_dict=None, param_ind=None):
        # load default model params from config file
        if mode == "default":
            model_params = self.default_params
        # load best model params after grid search
        elif mode == "best":
            self.compute_best_params()
            model_params = self.args.best_params
        # load one of possible configurations from grid search
        elif mode == "grid" and param_ind is not None:
            model_params = self.param_grid_list[param_ind]
        # load custom configuration
        elif mode == "custom" and param_dict is not None:
            model_params = param_dict
        else:
            raise ValueError("Provide a correct mode and a valid param_dict or param_ind.")

        self._set_args_attributes(model_params, f"{mode} model parameters applied, params:{model_params}")
        self.args.model_params = model_params

    
    def set_ids(self, mode="default", ids_dict=None, inner_fold=None):
        # load default outer cv ids
        if mode == "default":
            train_ids, val_ids, test_ids = load_fold_file(self.args)
            ids_dict = {"train": train_ids, "val": val_ids, "test": test_ids}
        # final training concatenating train and val
        elif mode == "final":       
            train_ids, val_ids, test_ids = load_fold_file(self.args) 
            ids_dict = {"train": np.concatenate([train_ids, val_ids]), "val": np.array([]), "test": test_ids}       
        # pass inner fold grid ids
        elif mode == "grid" and inner_fold is not None:
            self.args.inner_fold = inner_fold
            ids_dict = self.ids_grid_list[inner_fold]
        # use custom ids
        elif mode == "custom" and ids_dict is not None:
            pass
        else:
            raise ValueError("Provide a correct mode and a valid dict id.")

        # remove test ids in pretrain
        if self.args.train_mode == "pretrain":
            ids_dict["test"] = np.array([])
            
        self.args.cv_mode = mode
        self.args.ids = ids_dict
        self.args.logger.write(f'{mode} fold ids used.')
        self.args.logger.write(f"\n# train, val, test TS LOADED: {len(ids_dict['train'])}, {len(ids_dict['val'])}, {len(ids_dict['test'])}")


    def resolve_mode(self):
        # determine training mode and load checkpoint if provided.
        pretrainable_models = ("strats", "ehrmamba_lab")
        if self.args.model_type in pretrainable_models and self.args.load_ckpt_path is not None:
            self.args.pt_var_path = Path(self.args.load_ckpt_path) / 'pt_saved_variables.pkl'
            self.args.pt_dict_path = Path(self.args.load_ckpt_path) / 'checkpoint_best.bin'
            try:
                if not (self.args.pt_var_path.exists() and self.args.pt_dict_path.exists()):
                    raise FileNotFoundError("Required checkpoint files not found.")
                
                with open(self.args.pt_var_path, "rb") as f:
                    pickle.load(f)  # confirm readable
                with open(self.args.pt_dict_path, "rb") as f:
                    f.read(1)  # confirm readable (binary checkpoint, just touch)
                    
            except Exception as e:
                raise RuntimeError(f"Error loading checkpoint files: {e}")
            
            # set to finetune only if files can actually be read
            else:
                self.args.finetune = True
                self.args.train_mode = "finetune"

        elif self.args.model_type in pretrainable_models and self.args.pretrain:
            self.args.train_mode = "pretrain"

        else:
            # models different than strats can only be run in standard mode
            self.args.train_mode = "standard"

        if self.args.prefix is None:
            self.args.prefix = datetime.now().strftime("%d%m%y")
        
    def set_device(self):
        self.args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.args.model_type == "tcn":
            self.args.device = 'cpu' #TCN faster on CPU

    def setup_logging(self):
        self.args.logger = Logger(self.args.paths["output_path"], 'log.txt')
        self.args.logger.write('######### START #########')
        self.args.logger.write('Global environment loaded')
        self.args.logger.write(f'Training in {self.args.train_mode} mode on {self.args.device}')

    def set_stratify_batch(self):
        # if task is very unbalanced (NF) > stratify batch
        self.args.stratify_batch = 8 if "NF" in self.args.cohort else 0

    def get_param_grid_list(self):
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]
        self.param_grid_list = [dict(zip(keys, v)) for v in product(*values)]

    def get_ids_grid_list(self): # Generating new splits
        # read default fold
        train_ids, val_ids, _ = load_fold_file(self.args)
        outer_dev_ids = np.concatenate([train_ids, val_ids])  # remove test ids

        # read cohort file    
        cohort_file = pd.read_csv(self.args.paths["cohort_path"], compression='gzip')

        # extract relevant admissions, preserve order
        dev_adms = (
            cohort_file[cohort_file["hadm_id"].isin(outer_dev_ids)]
            .set_index("hadm_id").loc[outer_dev_ids]
            .reset_index()[["hadm_id","subject_id","label"]]
        )

        sgkf = StratifiedGroupKFold(n_splits=num_inner_folds, shuffle=True, random_state=self.args.seed)
        groups = dev_adms["subject_id"].values
        labels = dev_adms["label"].values
        admids = dev_adms["hadm_id"].values

        inner_splits = []

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(dev_adms, labels, groups=groups)):
            inner_splits.append({
                "train": admids[train_idx],
                "val": admids[val_idx],
                "test": np.array([])  # empty array for test
            }) 
        self.ids_grid_list = inner_splits
    
    def _get_param_names(self):

        # get best params names (from grid search or default params depending on training mode)
        if hasattr(self, "param_grid_list"):
            param_names = list(self.param_grid.keys())
        else:
            param_names = list(self.default_params.keys())
            
        if "mean_epochs" in param_names:
            param_names.remove("mean_epochs")

        return param_names

    def _check_grid_size(self, grid_res):
        # if trained in nested crossvalidation (grid = nested)
        if hasattr(self, "param_grid_list") and self.param_grid_list is not None:
            expected_len = num_inner_folds * len(self.param_grid_list)
            if len(grid_res) != expected_len:
                raise FileNotFoundError(
                    f"Grid results are incomplete! Expected {expected_len} rows, got {len(grid_res)}"
                )
        # if trained with train val test split (grid = best_epochs)
        else:
            if len(grid_res) != 1:
                raise FileNotFoundError(
                    f"Grid results are inconsistent! Expected 1 row, got {len(grid_res)}"
                )
            #if self.args.cv_mode != "grid" and len(grid_res) != num_inner_folds:
                #raise FileNotFoundError(f"Grid results are incomplete! Expected {num_inner_folds} rows, got {len(grid_res)}")
    
    def compute_best_params(self):

        # check if file exist and read
        grid_file = Path(self.args.paths["output_path"]) / "grid_results.csv"
        if not grid_file.exists():
            raise FileNotFoundError(f"Grid results file not found: {grid_file}") 
        grid_res = pd.read_csv(grid_file)    
        
        self._check_grid_size(grid_res)
        
        param_names = self._get_param_names()
            
        metric = "loss" if self.args.train_mode == "pretrain" else "auroc"
        ascending = self.args.train_mode == "pretrain"
        best_tuple = (
            grid_res
            .groupby(param_names, dropna=False)[[metric, "epoch"]]
            .mean()
            .sort_values(metric, ascending=ascending)
            .iloc[0]
        )
        # build dict of best params
        best_params = dict(zip(param_names, best_tuple.name))
        # ensure hid_dims is a list of ints if present
        best_params = self._check_param_list(best_params, ["hid_dims"])
            
        self.args.mean_epochs = best_tuple["epoch"]
        self.args.best_params = best_params
        self.args.logger.write(f"Best parameters selected: {self.args.best_params}")    
        
    def train(self):
        set_all_seeds(self.args.seed, self.args.fast)
        
        dataset = TimeSeriesDataset(self.args)
        model = build_model(self.args)       
        batcher = build_batcher(self.args, dataset.preproc.input_dict)
        
        # free dataset once batcher has extracted what it needs
        del dataset
        torch.cuda.empty_cache()
        
        trainer = Trainer(self.args, model, batcher)
        trainer.train() 
        
        # cleanup after training
        del model, batcher, trainer
        torch.cuda.empty_cache()     
        
    def train_full(self):
        
        # EXTRACT DATA if not available
        extractor = ExtractorTrain(self.args) if self.args.train_mode != "pretrain" else ExtractorPretrain(self.args)
            
        # GRID SEARCH > NESTED CROSSVALIDATION
        if self.args.grid == "nested":
            
            # get all inner folds and parameter combinations
            self.get_param_grid_list()
            self.get_ids_grid_list()
            
            # train on each parameter combination and each split
            for i, ids in enumerate(self.ids_grid_list):
                for p, params in enumerate(self.param_grid_list):        

                    self.set_model_params(mode="grid", param_ind=p)
                    self.set_ids(mode="grid", inner_fold=i)
                    self.train()

            # retrain with best params on the default split
            self.set_model_params(mode="best") 
            self.set_ids(mode="final")
            self.train()
            
        # GRID SEARCH > SIMPLE CROSSVALIDATION
        elif self.args.grid == "simple":

            # get all inner folds and parameter combinations
            self.get_param_grid_list()
            
            # train on each parameter combination and each split
            for p, params in enumerate(self.param_grid_list):        

                self.set_model_params(mode="grid", param_ind=p)
                self.set_ids(mode="default")
                self.train()
            # TO DO > ADD TRAINING ON WHOLE TRAINING SET (TRAIN + VAL)
            
        # SINGLE TRAINING  
        elif self.args.grid == "none":
            
            #train on default parameters and default split
            self.set_model_params(mode="default")
            self.set_ids(mode="default")
            self.train()
            
        # TRAIN ON THE BEST CONFIG FOUND in GRID-SEARCH (saved in YAML file) with predefined number of epoch
        elif self.args.grid == "best":
            
            #train on default parameters and default split
            self.set_model_params(mode="default") 
            self.set_ids(mode="final")
            self.train()

        elif self.args.grid == "best_epochs":

            #train on default parameters and default split 
            self.set_model_params(mode="default")
            self.set_ids(mode="default")
            self.args.cv_mode = "grid" # mode set to grid so that results are not stored as final
            self.train()
                
            # retrain with best params and number of epochs on the final split
            self.set_model_params(mode="best") 
            self.set_ids(mode="final")
            self.train()
            
        else:
            raise ValueError(
                f"Invalid value for --grid: {self.args.grid}. "
                "Expected one of: 'nested', 'simple', 'none', 'best', 'best_epochs'"
            )

            
    def train_best(self): # after CV
        
        # EXTRACT DATA if not available
        extractor = ExtractorTrain(self.args) if self.args.train_mode != "pretrain" else ExtractorPretrain(self.args)
            
        # get all inner folds and parameter combinations
        self.get_param_grid_list()
        self.get_ids_grid_list()

        # retrain with best params on the default split
        self.set_model_params(mode="best") 
        self.set_ids(mode="final")
        self.train()

            
            
"""           
    def get_ids_grid_list_(self): # Using existing splits # DEPRECATED
        all_ids = []

        # Load test IDs for all outer folds
        for r in range(self.nfolds):
            fold_file = self.args.paths["CV_folds_path"] / f'fold_{r}.pkl'
            with open(fold_file, 'rb') as f:
                _, _, test_ids = pickle.load(f)
            all_ids.append(test_ids)

        # Select outer dev IDs (all folds except the current outer fold)
        outer_dev_ids = [ids for i, ids in enumerate(all_ids) if i != self.args.fold]

        inner_splits = []
        for i, val_ids in enumerate(outer_dev_ids):
            # Combine remaining outer dev folds as inner train
            train_folds = [outer_dev_ids[j] for j in range(len(outer_dev_ids)) if j != i]
            train_ids = np.concatenate(train_folds)
            
            inner_splits.append({
                "train": train_ids,
                "val": val_ids,
                "test": np.array([])  # empty array for test
            })

        self.ids_grid_list = inner_splits

        # TRAIN ON THE BEST CONFIG FOUND in GRID-SEARCH (saved in YAML file) but train first w patience to determine number of epochs
        elif self.args.grid == "best_patience":
            
            # Train on default parameters and default split
            self.set_model_params(mode="default")
            self.set_ids(mode="default")
            self.args.final = False
            self.train()

            # Read the number of epochs from the validation results
            results_file = self.args.paths["output_path"] / "results_final.csv"
            rf = pd.read_csv(results_file)
            val_epochs = rf.loc[rf["split"] == "val", "epoch"].values
            ep = int(round(val_epochs.mean()))

            # Retrain on final split using mean number of epochs
            self.set_model_params(mode="default")
            self.set_ids(mode="final")
            self.args.mean_epochs = ep
            self.args.final = True
            self.train()

        elif self.args.grid == "best_epochs":

            # train in nested CV to find the best number of epochs
            self.get_ids_grid_list()

            self.set_model_params(mode="default")

            for i, ids in enumerate(self.ids_grid_list):
                self.set_ids(mode="grid", inner_fold=i)
                self.train()
                
            # retrain with best params on the default split
            self.set_model_params(mode="best") 
            self.set_ids(mode="final")
            self.train()
"""

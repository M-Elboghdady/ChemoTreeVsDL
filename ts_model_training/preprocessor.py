import os
import pickle
import numpy as np
import pandas as pd
from ts_model_training.utils import discrete_tensors, fill_impute, fill_mean, compute_means_stds_df, compute_deltas, compute_holdout
from pathlib import Path
class Preprocessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.args = dataset.args
        self.data = dataset.data
        self.train_ind = self.dataset.splits["train"]

    def set_variables(self):          
        self.variables = self.get_vars()
        self.var_to_ind = {v: i for i, v in enumerate(self.variables)}
        # remove vars not in pretrain
        self.data = self.data[self.data.itemid.isin(self.variables)]
        self.data["var_ind"] = self.data.itemid.map(self.var_to_ind)
        self.args.V = len(self.variables)
        self.args.logger.write('\nTemporal variables: ' + ', '.join(self.variables))

    def get_vars(self):
        return sorted(self.data.itemid.unique())

    def trim(self):
        raise NotImplementedError

    def compute_means_stds(self):
        raise NotImplementedError

    def normalise(self):
        raise NotImplementedError

    def prepare_inputs(self):
        raise NotImplementedError
    
    def save_inputs(self):
        # merge current input_dict with splits and ts_id_to_ind
        self.input_dict = {
            **self.input_dict,
            "demo_norm": self.dataset.demo,
            "demo_raw": self.dataset.demo_raw,
            "demo_means" : self.dataset.demo_means, 
            "demo_stds" : self.dataset.demo_stds,           
            "splits": self.dataset.splits,
            "ts_id_to_ind": self.dataset.ts_id_to_ind,
            "var_to_ind": self.var_to_ind
        } # if the task is supervised, save also the target
        if self.args.train_mode != "pretrain":
            self.input_dict["target"] = self.dataset.y
            
        # pickle dump only if not training in nested crossvalidation
        if not (self.args.cv_mode == "grid"): #self.args.grid == "nested" and 
            output_path = Path(self.dataset.args.paths["output_path"]) / "input_dict.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(self.input_dict, f)


class PreprocessorA(Preprocessor):

    # preprocessing params
    # args.agg_int (default 24)
    def trim(self):
        self.data = self.data.sort_values("minute")
        self.data['int'] = (self.data['minute'] // (60 * self.args.agg_int)).astype(int)
        self.args.T = self.data.int.max() + 1
        self.args.logger.write('\nData discretised')
        self.args.logger.write('# intervals: '+str(self.args.T))

    def normalise(self):
        self.means, self.stds = self.compute_means_stds()
        self.values = (self.values-self.means)/self.stds
        self.args.logger.write('Data normalised')

    def compute_means_stds(self):
        means = self.values[self.train_ind].mean(axis=(0, 1), keepdims=True)
        stds = self.values[self.train_ind].std(axis=(0, 1), keepdims=True)
        stds = np.where(stds == 0, 1.0, stds)
        return means, stds

    def prepare_inputs(self):

        ## DISCRETISATION
        self.set_variables()
        self.trim()
        last, avgs, self.obs, self.delta, sums, counts = discrete_tensors(self.data, self.args.N, self.args.T, self.args.V)

        ## AGGREGATION 
        if self.args.agg == "mean": # aggregation by average value
            self.values = avgs
        else: # default is aggregation by last recorded value
            self.values = last

        ## IMPUTATION
        if self.args.impute == "fill":
            self.values = fill_impute(self.values, self.obs)
        else: # default is mean imputation
            self.values = fill_mean(self.values, self.obs, self.train_ind)     
        self.input_dict = {"values_raw" : self.values, "obs" : self.obs, "delta" : self.delta}
        # compute means and stds
        self.normalise()
        self.input_dict.update({"values_norm": self.values, "values_means": self.means, "values_stds": self.stds})

        
        # VARIANTS (concatenate data)
        variant = self.args.variant
        if variant == "V":
            self.X = np.concatenate((self.values,), axis=-1)
        elif variant == "M":
            self.X = np.concatenate((self.obs,), axis=-1)
        elif variant == "D":
            self.X = np.concatenate((self.delta,), axis=-1)
        elif variant == "MD":
            self.X = np.concatenate((self.obs, self.delta), axis=-1)
        elif variant == "VM":
            self.X = np.concatenate((self.values, self.obs), axis=-1)
        elif variant == "VD":
            self.X = np.concatenate((self.values, self.delta), axis=-1)
        else:  # default: VMD
            self.X = np.concatenate((self.values, self.obs, self.delta), axis=-1)
        self.input_dict["X"] = self.X
        self.args.logger.write('Input prepared')


class PreprocessorB(Preprocessor):
    # preprocessing params
    # args.max_timesteps (default 1000)

    def trim(self):
        # eliminate duplicate lab measurements
        self.data = self.data.groupby(["hadm_id", "ts_ind", "itemid","var_ind","minute"]).value.mean().reset_index()
        if self.args.max_timesteps != -1:
            timestamps = self.data[['hadm_id', 'minute']].drop_duplicates().sample(frac=1)
            timestamps = timestamps.groupby('hadm_id').head(self.args.max_timesteps)
            self.data = self.data.merge(timestamps, on=['hadm_id', 'minute'], how='inner')
            self.args.logger.write('\nData trimmed to max length')
    
    def compute_means_stds(self):  
        return compute_means_stds_df(self.data, self.train_ind)

    def normalise(self):
        means_stds = self.compute_means_stds()
        self.data = self.data.merge(means_stds, on='itemid', how='left')
        self.data['value'] = (self.data['value'] - self.data['mean']) / self.data['std']
        self.args.logger.write('Data normalised')

    def prepare_inputs(self):
        model_type = self.args.model_type
        
        self.set_variables()
        self.trim()
        self.normalise()

        N = self.args.N
        V = self.args.V
        
        if model_type == 'grud':
            deltas = [[] for _ in range(N)]
        elif model_type == 'interpnet':
            times = [[] for _ in range(N)]
            holdout_masks = [[] for _ in range(N)]
        values = [[] for _ in range(N)]
        mask = [[] for _ in range(N)]

        for ts_ind, curr_data in self.data.groupby('ts_ind'):

            # get observed time points
            curr_times = sorted(curr_data.minute.unique())

            # construct value/mask matrices
            pivot_data = (
                curr_data
                .pivot(index="var_ind", columns="minute", values="value")
                .reindex(list(range(V)))  # consistent ordering
                .T
            )
            curr_values = pivot_data.fillna(0).to_numpy()
            curr_mask = pivot_data.notna().astype(int).to_numpy()

            # get deltas for grud model
            if model_type == 'grud':
                max_time = (self.args.days_before_discharge + 1) * 24 * 60  # minutes
                deltas[ts_ind] = compute_deltas(curr_times, curr_mask, max_time)

            # get masks for interpnet model
            elif model_type == 'interpnet':
                times[ts_ind] = list(np.array(curr_times) / 60)  # convert to hours
                curr_mask[0, :] = 1  # ensure at least one observation per feature
                holdout_masks[ts_ind] = compute_holdout(curr_mask)

            values[ts_ind] = curr_values
            mask[ts_ind] = curr_mask
            
        # save results to self
        self.values = values
        self.mask = mask
        self.input_dict = {"values" : self.values, "mask" : self.mask}
        if model_type == 'grud':
            self.deltas = deltas
            self.input_dict["deltas"] = self.deltas
        elif model_type == 'interpnet':
            self.times = times
            self.holdout_masks = holdout_masks
            self.input_dict["times"] = self.times
            self.input_dict["holdout_masks"] = self.holdout_masks
        self.args.logger.write('Input prepared')

class PreprocessorC(Preprocessor): # strats
    # preprocessing params
    # args.max_obs (default 1000) - can be set to -1 for no trimming
        
    def get_vars(self):
        raise NotImplementedError

    def compute_means_stds(self):
        raise NotImplementedError
            
    def normalise(self):
        means_stds = self.compute_means_stds()
        self.data = self.data.merge(means_stds, on='itemid', how='left')
        self.data['value'] = (self.data['value'] - self.data['mean']) / self.data['std']
        self.args.logger.write('Data normalised')
        
    def prepare_inputs(self):
        raise NotImplementedError
    
    
class PreprocessorC_unsup(PreprocessorC): # unsupervised
    
    def trim(self):
        # eliminate duplicate lab measurements
        self.data = self.data.groupby(["hadm_id", "ts_ind", "itemid","var_ind","minute"]).value.mean().reset_index()
        #self.data = self.data.sample(frac=1)
        #self.data = self.data.groupby('hadm_id').head(self.args.max_obs) # trimming is applied at batching stage
    
    def get_vars(self):
        self.pt_variables = sorted(self.data.itemid.unique())
        return self.pt_variables

    def compute_means_stds(self):     
        self.pt_means_stds = compute_means_stds_df(self.data, self.train_ind)
        return self.pt_means_stds

    def dump_stats(self):
        pt_var_path = os.path.join(self.args.paths["output_path"], 'pt_saved_variables.pkl')
        with open(pt_var_path, 'wb') as f:
            pickle.dump((self.pt_variables, self.pt_means_stds), f)

    def prepare_inputs(self):
        self.set_variables()
        self.trim()
        self.normalise()
        self.dump_stats()

        N = self.args.N
        values = [[] for _ in range(N)]
        times = [[] for _ in range(N)]
        varis = [[] for _ in range(N)]

        self.data = self.data.sample(frac=1).sort_values(by='minute')
        for row in self.data.itertuples():
            values[row.ts_ind].append(row.value)
            times[row.ts_ind].append(row.minute)
            varis[row.ts_ind].append(row.var_ind)
        self.values, self.times, self.varis = values, times, varis
        
        # unique sorted timestamps except the last one for each patient
        self.timestamps = [np.array(sorted(list(set(x)))[:-1]) for x in self.times]
        # only keep timepoints that occur 12h or later
        self.timestamps = [x[x>=720] for x in self.timestamps] 
        self.input_dict = {"values" : self.values, "times" : self.times, "varis": self.varis, "timestamps": self.timestamps}
        self.args.logger.write('Input prepared.')
        
        # get all admissions where there are no valid timestamps left after filtering
        delete = [i for i in range(self.args.N) if len(self.timestamps[i])==0]
        # remove from splits
        self.dataset.splits = {k:np.setdiff1d(v,delete) for k,v in self.dataset.splits.items()}    
        self.args.logger.write(str(len(delete)) + ' admissions removed.')   
        

class PreprocessorC_sup(PreprocessorC): # supervised

    def __init__(self, dataset):
        super().__init__(dataset)
        # If finetuning, load precomputed variables and normalization stats
        if self.args.train_mode == "finetune":
            self.pt_variables, self.pt_means_stds = pickle.load(open(self.args.pt_var_path, 'rb'))   
            
    def trim(self):
        # eliminate duplicate lab measurements
        self.data = self.data.groupby(["hadm_id", "ts_ind", "itemid","var_ind","minute"]).value.mean().reset_index()
        self.data = self.data.sample(frac=1)
        self.data = self.data.groupby('hadm_id').head(self.args.max_obs) 

    def get_vars(self):
        if self.args.train_mode == "finetune":  
            return self.pt_variables
        else:
            return sorted(self.data.itemid.unique())
        
    def compute_means_stds(self):  
        # strats can be pretrained and normalisation should use precomputed stats
        if self.args.train_mode == "finetune":
            return self.pt_means_stds
        else:
            return compute_means_stds_df(self.data, self.train_ind)

    def prepare_inputs(self):
        self.set_variables()
        self.trim()
        self.normalise()

        N = self.args.N
        values = [[] for _ in range(N)]
        times = [[] for _ in range(N)]
        varis = [[] for _ in range(N)]

        max_time = (self.args.days_before_discharge + 1) * 24 * 60
        self.data['minute'] = self.data['minute']/max_time*2-1

        for row in self.data.itertuples():
            values[row.ts_ind].append(row.value)
            times[row.ts_ind].append(row.minute)
            varis[row.ts_ind].append(row.var_ind)
        self.values, self.times, self.varis = values, times, varis
        self.input_dict = {"values" : self.values, "times" : self.times, "varis": self.varis}
        self.args.logger.write('Input prepared')


class PreprocessorEHRMamba(Preprocessor):
    """Event-level lab sequences for ehrmamba_lab (no V/M/D discretization)."""

    PAD_ID = 0
    CLS_ID = 1
    REG_ID = 2
    UNK_ID = 3
    LAB_ID_OFFSET = 4

    def prepare_inputs(self):
        args = self.args
        logger = args.logger

        # Exact minute timestamps required for continuous Time2Vec event_times
        if getattr(args, "drop_minutes", False):
            raise ValueError(
                "ehrmamba_lab requires drop_minutes=False "
                "(do not pass --drop_minutes)."
            )

        max_seq_len = int(getattr(args, "max_seq_len", 512))
        if max_seq_len < 3:
            raise ValueError(f"max_seq_len must be >= 3, got {max_seq_len}")

        days = int(getattr(args, "days_before_discharge", 14))
        max_minutes = (days + 1) * 24 * 60

        self.variables = sorted(self.data.itemid.astype(str).unique())
        self.var_to_ind = {v: i + self.LAB_ID_OFFSET for i, v in enumerate(self.variables)}

        vocab_size = self.LAB_ID_OFFSET + len(self.variables)
        args.vocab_size = vocab_size
        args.V = len(self.variables)

        logger.write(
            f"\nEHRMamba-Lab vocab_size={vocab_size} "
            f"(#labs={len(self.variables)}) days={days} "
            f"max_minutes={max_minutes}"
        )

        data = self.data.copy()

        if "hadm_id" not in data.columns:
            raise ValueError("EHRMamba-Lab requires hadm_id in the event table")

        data["itemid"] = data["itemid"].astype(str)

        data = (
            data.groupby(
                ["hadm_id", "ts_ind", "minute", "itemid"],
                as_index=False
            )["value"]
            .mean()
        )

        data["event_id"] = (
            data["itemid"]
            .map(self.var_to_ind)
            .fillna(self.UNK_ID)
            .astype(int)
        )

        if data.empty:
            raise ValueError("No temporal lab events remain after filtering.")

        logger.write(
            f"  minute range: min={int(data['minute'].min())} "
            f"max={int(data['minute'].max())} "
            f"valid_range=[0, {max_minutes - 1}]"
        )

        if (data["minute"] < 0).any():
            raise ValueError("Found negative event times (minute < 0).")

        if (data["minute"] >= max_minutes).any():
            raise ValueError(
                f"Found events outside valid temporal range "
                f"(minute >= {max_minutes})."
            )

        if not np.isfinite(data["value"].to_numpy(dtype=float)).all():
            raise ValueError("Found non-finite raw lab values")

        train_rows = data[data["ts_ind"].isin(self.train_ind)]

        if train_rows.empty:
            raise ValueError("No training lab events available for normalization.")

        stats = (
            train_rows.groupby("itemid")["value"]
            .agg(["mean", "std"])
            .reindex(self.variables)
        )

        means = stats["mean"].fillna(0.0).to_dict()
        stds_raw = stats["std"].replace(0, np.nan).fillna(1.0).to_dict()
        stds = {k: (v if v > 1e-6 else 1.0) for k, v in stds_raw.items()}

        def norm_value(itemid, val):
            return (val - means.get(itemid, 0.0)) / stds.get(itemid, 1.0)

        data["value_norm"] = [
            norm_value(i, v)
            for i, v in zip(data["itemid"], data["value"])
        ]

        if not np.isfinite(data["value_norm"].to_numpy(dtype=float)).all():
            raise ValueError("Found non-finite normalized lab values")

        data["event_time"] = data["minute"].astype(np.float32) / 60.0

        if not np.isfinite(data["event_time"].to_numpy(dtype=float)).all():
            raise ValueError("Found non-finite continuous event times")

        N = int(args.N)

        if (data["ts_ind"] < 0).any() or (data["ts_ind"] >= N).any():
            raise ValueError(f"ts_ind out of range [0, {N})")

        event_ids = np.zeros((N, max_seq_len), dtype=np.int64)
        values = np.zeros((N, max_seq_len), dtype=np.float32)
        event_mask = np.zeros((N, max_seq_len), dtype=np.float32)
        event_times = np.zeros((N, max_seq_len), dtype=np.float32)
        lengths = np.zeros(N, dtype=np.int64)

        counts = np.zeros(N, dtype=np.int64)
        removed = np.zeros(N, dtype=np.int64)
        n_trunc = 0

        grouped = (
            data.sort_values(
                ["ts_ind", "minute", "itemid"],
                kind="stable"
            )
            .groupby("ts_ind", sort=False)
        )

        for ts_ind, g in grouped:
            ts_ind = int(ts_ind)

            ev = g[
                ["event_id", "value_norm", "event_time"]
            ].to_numpy()

            n_events = len(ev)
            counts[ts_ind] = n_events
            capacity = max_seq_len - 2

            if n_events > capacity:
                n_trunc += 1
                removed[ts_ind] = n_events - capacity
                ev = ev[-capacity:]

            seq_event = [self.CLS_ID] + ev[:, 0].astype(int).tolist() + [self.REG_ID]
            seq_val = [0.0] + ev[:, 1].astype(float).tolist() + [0.0]
            seq_time = [0.0] + ev[:, 2].astype(float).tolist() + [0.0]
            seq_event_mask = [0.0] + [1.0] * len(ev) + [0.0]

            L = len(seq_event)

            lengths[ts_ind] = L
            event_ids[ts_ind, :L] = seq_event
            values[ts_ind, :L] = seq_val
            event_times[ts_ind, :L] = seq_time
            event_mask[ts_ind, :L] = seq_event_mask

        for ts_ind in np.where(lengths == 0)[0]:
            event_ids[ts_ind, 0] = self.CLS_ID
            event_ids[ts_ind, 1] = self.REG_ID
            lengths[ts_ind] = 2

        if lengths.min() < 2 or lengths.max() > max_seq_len:
            raise ValueError(
                f"Invalid lengths: min={lengths.min()} "
                f"max={lengths.max()} "
                f"max_seq_len={max_seq_len}"
            )

        reg_ok = np.all(
            event_ids[np.arange(N), lengths - 1] == self.REG_ID
        )

        if not reg_ok:
            raise ValueError("Some sequences do not end with REG at lengths-1")

        def pct(p):
            return float(np.percentile(counts, p))

        frac_trunc = n_trunc / float(N)

        mean_removed_truncated = (
            float(removed[removed > 0].mean())
            if np.any(removed > 0)
            else 0.0
        )

        logger.write(
            "\nEHRMamba-Lab event counts per admission "
            "(labs only, excluding CLS/REG):"
        )

        logger.write(
            f"  N={N} min={counts.min()} "
            f"median={np.median(counts):.0f} "
            f"p75={pct(75):.0f} "
            f"p90={pct(90):.0f} "
            f"p95={pct(95):.0f} "
            f"p99={pct(99):.0f} "
            f"max={counts.max()}"
        )

        logger.write(
            f"  provisional max_seq_len={max_seq_len}"
        )

        logger.write(
            f"  truncated={n_trunc} "
            f"({100 * frac_trunc:.2f}% of N) "
            f"mean_removed_all={removed.mean():.2f} "
            f"mean_removed_truncated={mean_removed_truncated:.2f} "
            f"max_removed={int(removed.max())}"
        )

        if frac_trunc >= 0.05:
            logger.write(
                "  WARNING: truncation >= 5%; "
                "revise max_seq_len before full HPO"
            )

        self.input_dict = {
            "event_ids": event_ids,
            "values": values,
            "event_mask": event_mask,
            "event_times": event_times,
            "lengths": lengths,
            "lab_means": means,
            "lab_stds": stds,
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
        }

        logger.write(
            f"  tensors: event_ids{event_ids.shape} "
            f"values{values.shape} "
            f"event_mask{event_mask.shape} "
            f"event_times{event_times.shape} "
            f"lengths{lengths.shape}"
        )

        logger.write(
            "EHRMamba-Lab event sequences prepared (no V/M/D)."
        )

import os
import pickle
import numpy as np
import pandas as pd
from ts_model_training.utils import discrete_tensors, fill_impute, fill_mean, compute_means_stds_df, compute_deltas, compute_holdout
from pathlib import Path
from config.constants import PROJECT_ROOT, MIMIC_DIR
import gc
import shutil
import pickle

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
        if not (self.args.cv_mode == "grid") and self.args.model_type != "ehrmamba_lab": #self.args.grid == "nested" and
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

        if self.args.model_type == "ehrmamba_lab" and self.args.pretrain_protocol == "strats_matched":
            variables = [str(v) for v in self.pt_variables]
            stats = self.pt_means_stds.copy()
            stats.index = stats.index.astype(str)

            payload = {
                "variables": variables,
                "var_to_ind": {v: i + 4 for i, v in enumerate(variables)},
                "lab_means": stats["mean"].to_dict(),
                "lab_stds": stats["std"].to_dict(),
                "lab_bin_edges": {},
                "value_encoding": "continuous",
                "vocab_size": len(variables) + 4,
            }
        else:
            payload = (self.pt_variables, self.pt_means_stds)

        with open(pt_var_path, 'wb') as f:
            pickle.dump(payload, f)

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
    N_VALUE_BINS = 5

    def prepare_inputs(self):
        args = self.args
        logger = args.logger

        self.value_encoding = str(getattr(args, "value_encoding", "continuous")).lower()
        args.value_encoding = self.value_encoding
        if self.value_encoding not in ("continuous", "bins5"):
            raise ValueError(f"Unknown value_encoding: {self.value_encoding}")

        finetune = args.train_mode == "finetune"
        pt_means, pt_stds, pt_bin_edges = {}, {}, {}
        if finetune:
            with open(args.pt_var_path, "rb") as f:
                payload = pickle.load(f)
            self.value_encoding = str(payload["value_encoding"]).lower()
            args.value_encoding = self.value_encoding
            self.variables = list(payload["variables"])
            self.var_to_ind = dict(payload["var_to_ind"])
            pt_means = dict(payload.get("lab_means") or {})
            pt_stds = dict(payload.get("lab_stds") or {})
            pt_bin_edges = dict(payload.get("lab_bin_edges") or {})
            vocab_size = int(payload["vocab_size"])
            args.vocab_size = vocab_size
            args.V = len(self.variables)
            logger.write(
                f"\nEHRMamba-Lab FINETUNE: reused pt vocab_size={vocab_size} "
                f"(#labs={len(self.variables)}) encoding={self.value_encoding}"
            )

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

        if not finetune:
            self.variables = sorted(self.data.itemid.astype(str).unique())
            self.var_to_ind = {v: i + self.LAB_ID_OFFSET for i, v in enumerate(self.variables)}

            vocab_size = self.LAB_ID_OFFSET + len(self.variables)
            args.vocab_size = vocab_size
            args.V = len(self.variables)

            logger.write(
                f"\nEHRMamba-Lab vocab_size={vocab_size} "
                f"(#labs={len(self.variables)}) days={days} "
                f"max_minutes={max_minutes} value_encoding={self.value_encoding}"
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

        if finetune:
            data = data[data["itemid"].isin(self.variables)]

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

        if (not finetune) and train_rows.empty:
            raise ValueError("No training lab events available for value encoding.")

        means = {}
        stds = {}
        bin_edges = {}

        if self.value_encoding == "continuous":
            if finetune:
                means, stds = pt_means, pt_stds
            else:
                stats = (
                    train_rows.groupby("itemid")["value"]
                    .agg(["mean", "std"])
                    .reindex(self.variables)
                )
                means = stats["mean"].fillna(0.0).to_dict()
                stds_raw = stats["std"].replace(0, np.nan).fillna(1.0).to_dict()
                stds = {k: (v if v > 1e-6 else 1.0) for k, v in stds_raw.items()}

            def encode_value(itemid, val):
                return (val - means.get(itemid, 0.0)) / stds.get(itemid, 1.0)

            data["value_encoded"] = [
                encode_value(i, v)
                for i, v in zip(data["itemid"], data["value"])
            ]
            if not np.isfinite(data["value_encoded"].to_numpy(dtype=float)).all():
                raise ValueError("Found non-finite normalized lab values")

        else:
            if finetune:
                bin_edges = pt_bin_edges
            else:
                qs = [0.2, 0.4, 0.6, 0.8]
                for itemid, g in train_rows.groupby("itemid"):
                    edges = np.quantile(g["value"].to_numpy(dtype=float), qs).astype(float)
                    bin_edges[itemid] = edges

                for itemid in self.variables:
                    if itemid not in bin_edges:
                        bin_edges[itemid] = np.array([], dtype=float)

            def encode_value(itemid, val):
                edges = bin_edges[itemid]
                if len(edges) == 0:
                    return 0
                return int(np.searchsorted(edges, val, side="right"))

            data["value_encoded"] = [
                encode_value(i, v)
                for i, v in zip(data["itemid"], data["value"])
            ]
            bins = data["value_encoded"].to_numpy(dtype=int)
            if (bins < 0).any() or (bins >= self.N_VALUE_BINS).any():
                raise ValueError(
                    f"value bins out of range [0, {self.N_VALUE_BINS - 1}]"
                )

        data["event_time"] = data["minute"].astype(np.float32) / 60.0

        if not np.isfinite(data["event_time"].to_numpy(dtype=float)).all():
            raise ValueError("Found non-finite continuous event times")

        N = int(args.N)

        if (data["ts_ind"] < 0).any() or (data["ts_ind"] >= N).any():
            raise ValueError(f"ts_ind out of range [0, {N})")

        event_ids = np.zeros((N, max_seq_len), dtype=np.int64)
        if self.value_encoding == "continuous":
            values = np.zeros((N, max_seq_len), dtype=np.float32)
        else:
            values = np.zeros((N, max_seq_len), dtype=np.int64)
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
                ["event_id", "value_encoded", "event_time"]
            ].to_numpy()

            n_events = len(ev)
            counts[ts_ind] = n_events
            capacity = max_seq_len - 2

            if n_events > capacity:
                n_trunc += 1
                removed[ts_ind] = n_events - capacity
                ev = ev[-capacity:]

            seq_event = [self.CLS_ID] + ev[:, 0].astype(int).tolist() + [self.REG_ID]
            if self.value_encoding == "continuous":
                seq_val = [0.0] + ev[:, 1].astype(float).tolist() + [0.0]
            else:
                seq_val = [0] + ev[:, 1].astype(int).tolist() + [0]
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
            "lab_bin_edges": bin_edges,
            "value_encoding": self.value_encoding,
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

class PreprocessorEHRMamba_unsup(PreprocessorEHRMamba):
    """Prepares EHRMamba-Lab inputs for future-value pretraining."""

    def dump_pt_stats(self):
        """Saves vocabulary and normalization metadata."""
        pt_var_path = os.path.join(self.args.paths["output_path"], "pt_saved_variables.pkl")
        payload = {
            "variables": list(self.variables),
            "var_to_ind": dict(self.var_to_ind),
            "lab_means": self.input_dict["lab_means"],
            "lab_stds": self.input_dict["lab_stds"],
            "lab_bin_edges": self.input_dict.get("lab_bin_edges", {}),
            "value_encoding": self.input_dict["value_encoding"],
            "vocab_size": int(self.input_dict["vocab_size"]),
        }
        with open(pt_var_path, "wb") as f:
            pickle.dump(payload, f)
        self.args.logger.write(f"Saved EHRMamba pretrain metadata → {pt_var_path}")

    def _valid_cuts_for_admission(self, i, forecast_h, pred_h, min_cut_h):
        """Returns valid forecasting cutoff times."""
        L = int(self.input_dict["lengths"][i])
        if L < 3:
            return np.array([], dtype=np.float32)

        pos = np.arange(1, L - 1, dtype=np.int64)
        pos = pos[self.input_dict["event_mask"][i, pos] > 0.5]
        if pos.size == 0:
            return np.array([], dtype=np.float32)

        times_h = self.input_dict["event_times"][i, pos].astype(np.float32)
        uniq = np.unique(times_h)
        if uniq.size < 2:
            return np.array([], dtype=np.float32)

        cands = []
        for t1 in uniq[:-1]:
            if t1 < min_cut_h:
                continue
            n_past = np.sum((times_h < t1) & (times_h >= t1 - forecast_h))
            n_fut = np.sum((times_h >= t1) & (times_h < t1 + pred_h))
            if n_past > 0 and n_fut > 0:
                cands.append(float(t1))
        return np.asarray(cands, dtype=np.float32)

    def prepare_inputs(self):
        """Adds forecasting cutoffs and saves pretraining metadata."""
        enc = str(getattr(self.args, "value_encoding", "continuous")).lower()
        if enc != "continuous":
            raise ValueError(
                "PreprocessorEHRMamba_unsup requires value_encoding='continuous' "
                f"(got '{enc}')."
            )

        self.args.value_encoding = "continuous"
        super().prepare_inputs()

        forecast_h = float(self.args.window_forecast) / 60.0
        pred_h = float(self.args.window_pred) / 60.0
        min_cut_h = 720.0 / 60.0
        N = int(self.args.N)

        timestamps = [
            self._valid_cuts_for_admission(i, forecast_h, pred_h, min_cut_h)
            for i in range(N)
        ]
        self.input_dict["timestamps"] = timestamps

        delete = [i for i in range(N) if len(timestamps[i]) == 0]
        self.dataset.splits = {
            k: np.setdiff1d(v, delete) for k, v in self.dataset.splits.items()
        }
        self.args.logger.write(
            f"EHRMamba pretrain: removed {len(delete)} admissions with no valid cutoff."
        )

        self.dump_pt_stats()


class PreprocessorEHRMambaPaper(Preprocessor):
    SPECIAL_TOKENS=["[PAD]","[UNK]","[MASK]","[CLS]","[VS]","[VE]","[REG]",
                    "[W0]","[W1]","[W2]","[W3]","[M0]","[M1]","[M2]","[M3]",
                    "[M4]","[M5]","[M6]","[M7]","[M8]","[M9]","[M10]","[M11]",
                    "[M12]","[LT]"]
    N_BUCKETS=64
    COLS=["subject_id","hadm_id","event_time","concept_id","lab_itemid","lab_value"]

    def __init__(self,dataset):
        super().__init__(dataset)
        self.mimic_dir=Path(MIMIC_DIR)
        self.pt_meta=pickle.load(open(self.args.pt_var_path,"rb")) if self.args.train_mode!="pretrain" else None

    def _event_cache_dir(self):
        return Path(self.args.paths["saved_data_path"])/"ehrmamba_paper_event_cache"/self.args.cohort

    def _parse_time(self,x):
        return pd.to_datetime(x,format="mixed")

    def _write_events(self,x):
        if x.empty:return
        x=x[self.COLS].copy()
        x["_b"]=x.subject_id.astype(np.int64)%self.N_BUCKETS
        for b,g in x.groupby("_b",sort=False):
            p=self._cache/f"bucket_{int(b):02d}.csv.gz"
            g[self.COLS].to_csv(p,mode="a",index=False,header=not p.exists(),
                                compression={"method":"gzip","compresslevel":1})

    def _build_event_cache(self,subjects):
        self._cache=self._event_cache_dir()
        ready,sf=self._cache/"READY",self._cache/"subjects.pkl"
        subjects=set(map(int,subjects))

        if ready.exists() and sf.exists() and pickle.load(open(sf,"rb"))==subjects:
            self.args.logger.write(f"\nUsing event cache: {self._cache}")
            return

        if self._cache.exists():shutil.rmtree(self._cache)
        self._cache.mkdir(parents=True)

        for x in pd.read_csv(self.mimic_dir/"hosp/labevents.csv.gz",
                             usecols=["subject_id","hadm_id","itemid","charttime","valuenum"],
                             chunksize=1_000_000):
            x=x[x.subject_id.isin(subjects)&x.charttime.notna()&x.valuenum.notna()]
            if x.empty:continue
            y=x[["subject_id","hadm_id"]].copy()
            y["event_time"]=x.charttime
            y["concept_id"]="LAB_"+x.itemid.astype(str)
            y["lab_itemid"]=x.itemid.astype(str)
            y["lab_value"]=x.valuenum.astype(float)
            self._write_events(y)
        self.args.logger.write("\nCached labs")

        for x in pd.read_csv(self.mimic_dir/"hosp/prescriptions.csv.gz",
                             usecols=["subject_id","hadm_id","starttime","stoptime","drug","gsn"],
                             chunksize=1_000_000,dtype={"drug":"string","gsn":"string"}):
            x=x[x.subject_id.isin(subjects)&x.drug.notna()].copy()
            if x.empty:continue
            drug=x.drug.astype(str).str.strip()
            gsn=x.gsn.fillna("UNK").astype(str)
            for col,state in [("starttime","STARTED"),("stoptime","STOPPED")]:
                m=x[col].notna()
                if not m.any():continue
                y=x.loc[m,["subject_id","hadm_id"]].copy()
                y["event_time"]=x.loc[m,col]
                y["concept_id"]="MEDICATION//"+state+"//"+drug.loc[m]+"//"+gsn.loc[m]
                y["lab_itemid"],y["lab_value"]=pd.NA,np.nan
                self._write_events(y)
        self.args.logger.write("\nCached medications")

        for x in pd.read_csv(self.mimic_dir/"hosp/procedures_icd.csv.gz",
                             usecols=["subject_id","hadm_id","chartdate","icd_code","icd_version"],
                             chunksize=1_000_000,dtype={"icd_code":"string"}):
            x=x[x.subject_id.isin(subjects)&x.icd_code.notna()&x.chartdate.notna()].copy()
            if x.empty:continue
            y=x[["subject_id","hadm_id"]].copy()
            y["event_time"]=x.chartdate
            y["concept_id"]="PROCEDURE//ICD//"+x.icd_version.astype(str)+"//"+x.icd_code.astype(str)
            y["lab_itemid"],y["lab_value"]=pd.NA,np.nan
            self._write_events(y)
        self.args.logger.write("\nCached procedures")

        pickle.dump(subjects,open(sf,"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        ready.touch()

    def _load_admissions(self,subjects):
        x=pd.read_csv(self.mimic_dir/"hosp/admissions.csv.gz",
                      usecols=["subject_id","hadm_id","admittime","dischtime"])
        x=x[x.subject_id.isin(subjects)].copy()
        x["admittime"]=pd.to_datetime(x.admittime)
        x["dischtime"]=pd.to_datetime(x.dischtime)
        return {int(s):(g.admittime.to_numpy("datetime64[ns]"),
                        g.dischtime.to_numpy("datetime64[ns]"),
                        g.hadm_id.to_numpy())
                for s,g in x.groupby("subject_id",sort=False)}

    def _load_ages(self,subjects):
        x=pd.read_csv(self.mimic_dir/"hosp/patients.csv.gz",
                      usecols=["subject_id","anchor_age","anchor_year"])
        return {int(r.subject_id):(float(r.anchor_age),int(r.anchor_year))
                for r in x[x.subject_id.isin(subjects)].itertuples(index=False)}

    def _attach_hadm(self,g,admissions):
        m=g.hadm_id.isna()
        if not m.any():return g
        a=admissions.get(int(g.subject_id.iloc[0]))
        if a is None:return g
        t=g.loc[m,"event_time"].to_numpy("datetime64[ns]")
        start,end,ids=a
        match=(t[:,None]>=start)&(t[:,None]<=end)
        ok=match.any(axis=1)
        if ok.any():
            j=match.argmax(axis=1)
            idx=g.index[m].to_numpy()
            g.loc[idx[ok],"hadm_id"]=ids[j[ok]]
        return g

    def _encode_labs(self,x,edges):
        m=x.lab_itemid.notna()
        if not m.any():return x
        ids=x.loc[m,"lab_itemid"].astype(str)
        vals=x.loc[m,"lab_value"].astype(float)
        x.loc[m,"concept_id"]=[
            f"LAB_{i}//BIN{np.searchsorted(edges[i],v,side='right')}"
            if i in edges else f"LAB_{i}//BINUNK"
            for i,v in zip(ids,vals)
        ]
        return x

    def assign_visits(self,events):
        x=events.sort_values("event_time",kind="stable").copy()
        x["visit_order"]=pd.factorize(x.hadm_id,sort=False)[0]+1
        x["visit_segment"]=((x.visit_order-1)%2)+1
        return x

    def time_token(self,t1,t2):
        d=max(0.,(t2-t1).total_seconds()/86400.)
        if d<28:return f"[W{min(3,int(d//7))}]"
        m=int(d//30.44)
        return f"[M{m}]" if m<=12 else "[LT]"

    def build_sequence(self,events):
        x=self.assign_visits(events)
        seq=[{"token":"[CLS]","visit_order":0,"visit_segment":0,"event_time":pd.NaT}]
        prev=None

        for vo,g in x.groupby("visit_order",sort=True):
            g=g.sort_values("event_time",kind="stable")
            seg=int(g.visit_segment.iloc[0])
            start,end=g.event_time.iloc[0],g.event_time.iloc[-1]

            if prev is not None:
                seq.append({"token":self.time_token(prev,start),"visit_order":0,
                            "visit_segment":0,"event_time":pd.NaT})

            seq.append({"token":"[VS]","visit_order":0,"visit_segment":0,"event_time":pd.NaT})
            seq.extend({"token":t,"visit_order":int(vo),"visit_segment":seg,"event_time":et}
                       for t,et in zip(g.concept_id,g.event_time))
            seq.extend([
                {"token":"[VE]","visit_order":0,"visit_segment":0,"event_time":pd.NaT},
                {"token":"[REG]","visit_order":0,"visit_segment":0,"event_time":pd.NaT}
            ])
            prev=end

        return pd.DataFrame(seq)

    def add_age(self,x,sid,ages):
        x=x.copy()
        x["age"]=0.
        if sid not in ages:return x
        age,year=ages[sid]
        m=x.event_time.notna()
        x.loc[m,"age"]=(age+x.loc[m,"event_time"].dt.year-year).astype(np.float32)
        return x

    def add_time(self,x):
        x=x.copy()
        x["time"]=0.
        m=x.event_time.notna()
        if not m.any():return x
        t0=x.loc[m,"event_time"].min()
        x.loc[m,"time"]=((x.loc[m,"event_time"]-t0).dt.total_seconds()/(86400.*7.)).astype(np.float32)
        return x

    def add_type_ids(self,x):
        def f(t):
            if t=="[PAD]":return 0
            if t.startswith("LAB_"):return 1
            if t.startswith("MEDICATION//"):return 2
            if t.startswith("PROCEDURE//"):return 3
            if t=="[CLS]":return 4
            if t in ("[VS]","[VE]"):return 5
            if t=="[REG]":return 6
            if t.startswith("[W") or t.startswith("[M") or t=="[LT]":return 7
            return 8

        x=x.copy()
        x["type_id"]=x.token.map(f).astype(np.int8)
        return x

    def add_token_ids(self,x,vocab):
        x=x.copy()
        x["token_id"]=x.token.map(vocab).fillna(vocab["[UNK]"]).astype(np.int32)
        return x

    def _make_sequence(self,g,sid,ages,bins,vocab):
        x=self.build_sequence(self._encode_labs(g.copy(),bins))
        x=self.add_age(x,sid,ages)
        x=self.add_time(x)
        x=self.add_type_ids(x)
        return self.add_token_ids(x,vocab)

    def _compact(self,x):
        return x[["token_id","type_id","age","time","visit_order","visit_segment"]].astype({
            "token_id":np.int32,"type_id":np.int8,"age":np.float32,"time":np.float32,
            "visit_order":np.int32,"visit_segment":np.int8
        })

    def _chunk(self,x):
        L=int(self.args.max_seq_len)
        out=[]
        start=0
        while start<len(x):
            end=min(start+L,len(x))
            out.append(self._compact(x.iloc[start:end].reset_index(drop=True)))
            if end==len(x):break
            start=end-1
        return out

    def _truncate_finetune(self,x):
        L=int(self.args.max_seq_len)
        if len(x)<=L:return self._compact(x)
        return self._compact(pd.concat([x.iloc[[0]],x.iloc[-(L-1):]],ignore_index=True))

    def _empty_sequence(self,vocab):
        x=pd.DataFrame([{"token":"[CLS]","visit_order":0,"visit_segment":0,"event_time":pd.NaT}])
        x["age"],x["time"]=0.,0.
        return self._compact(self.add_token_ids(self.add_type_ids(x),vocab))

    def _target_rows(self):
        c=self.dataset.cohort.copy()
        if "dischtime" not in c.columns:
            c=pd.read_csv(self.args.paths["cohort_path"],compression="gzip",parse_dates=["dischtime"])
        c["dischtime"]=pd.to_datetime(c.dischtime)
        ids=set(map(int,self.dataset.ts_id_to_ind))
        return c[c.hadm_id.isin(ids)][["subject_id","hadm_id","dischtime"]].drop_duplicates("hadm_id")

    def prepare_inputs(self):
        if self.args.train_mode=="pretrain":
            raise RuntimeError("Use PreprocessorEHRMambaPaperPretrain.")

        vocab,bins=self.pt_meta["vocab"],self.pt_meta["lab_bin_edges"]
        self.args.max_seq_len=int(self.pt_meta["max_seq_len"])
        max_pretrain_visits=int(self.pt_meta["max_num_visits"])

        targets=self._target_rows()
        subjects=set(targets.subject_id.astype(int))
        self._build_event_cache(subjects)

        admissions=self._load_admissions(subjects)
        ages=self._load_ages(subjects)
        sequences=[None]*len(self.dataset.ts_id_to_ind)
        max_visit=before=after=dropped=0

        for b in range(self.N_BUCKETS):
            p=self._cache/f"bucket_{b:02d}.csv.gz"

            if p.exists():
                e=pd.read_csv(p,dtype={"concept_id":"string","lab_itemid":"string"})
                e["event_time"]=self._parse_time(e.event_time)
                e["hadm_id"]=pd.to_numeric(e.hadm_id,errors="coerce").astype("Int64")
                by_sid={int(s):g for s,g in e.groupby("subject_id",sort=False)}
            else:
                e=pd.DataFrame(columns=self.COLS)
                by_sid={}

            rows=targets[targets.subject_id.astype(np.int64)%self.N_BUCKETS==b]

            for r in rows.itertuples(index=False):
                sid,hadm=int(r.subject_id),int(r.hadm_id)
                g=by_sid.get(sid)

                if g is not None:
                    start=r.dischtime-pd.DateOffset(days=int(self.args.days_before_discharge))
                    g=g[(g.event_time>=start)&(g.event_time<=r.dischtime)].copy()

                if g is None or g.empty:
                    x=self._empty_sequence(vocab)
                else:
                    before+=int(g.hadm_id.isna().sum())
                    g=self._attach_hadm(g,admissions)
                    m=int(g.hadm_id.isna().sum())
                    after+=m
                    dropped+=m
                    g=g[g.hadm_id.notna()].copy()

                    if g.empty:
                        x=self._empty_sequence(vocab)
                    else:
                        x=self._make_sequence(g,sid,ages,bins,vocab)
                        max_visit=max(max_visit,int(x.visit_order.max()))
                        x=self._truncate_finetune(x)

                sequences[self.dataset.ts_id_to_ind[hadm]]=x

            del e,by_sid
            gc.collect()

        n_missing=sum(x is None for x in sequences)
        if n_missing:
            empty=self._empty_sequence(vocab)
            sequences=[empty.copy() if x is None else x for x in sequences]

        if max_visit>max_pretrain_visits:
            raise ValueError(f"Fine-tune max visits {max_visit} > pretrained {max_pretrain_visits}")

        self.args.V=self.args.vocab_size=len(vocab)
        self.args.N=len(sequences)
        self.args.max_num_visits=max_pretrain_visits
        self.var_to_ind=vocab

        self.input_dict={
            "ehrmamba_sequences":sequences,
            "vocab":vocab,
            "vocab_size":len(vocab),
            "lab_bin_edges":bins,
            "max_seq_len":self.args.max_seq_len,
            "max_num_visits":max_pretrain_visits
        }

        self.args.logger.write(
            f"\nEHRMamba-Paper FINETUNE: N={len(sequences)}, vocab={len(vocab)}, "
            f"missing_hadm={before}->{after}, dropped={dropped}, "
            f"missing_sequences={n_missing}, max_visits={max_visit}"
        )


class PreprocessorEHRMambaPaperPretrain(PreprocessorEHRMambaPaper):

    def _get_subject_splits(self):
        c=self.dataset.cohort[["subject_id","hadm_id"]].drop_duplicates()
        out={
            s:set(c.loc[c.hadm_id.isin(set(map(int,self.args.ids.get(s,[])))),"subject_id"].astype(int))
            for s in ["train","val","test"]
        }

        assert out["train"].isdisjoint(out["val"])
        assert out["train"].isdisjoint(out["test"])
        assert out["val"].isdisjoint(out["test"])

        self.args.logger.write(
            f"\nEHRMamba pretrain subjects: train={len(out['train'])}, "
            f"val={len(out['val'])}, test={len(out['test'])}"
        )
        self.args.logger.write(
            f"\nPretrain cohort admissions={len(c)}, subjects={c.subject_id.nunique()}"
        )
        return out

    def _seq_cache_dir(self):
        return Path(self.args.paths["saved_data_path"])/"ehrmamba_paper_sequence_cache"/\
               self.args.cohort/f"fold_{self.args.fold}_L{int(self.args.max_seq_len)}"

    def _prepare_seq_cache(self,ss):
        self._seq_cache=self._seq_cache_dir()
        p=self._seq_cache/"subjects.pkl"
        current={k:set(map(int,ss[k])) for k in ["train","val","test"]}

        if self._seq_cache.exists() and p.exists() and pickle.load(open(p,"rb"))==current:
            self.args.logger.write(f"\nUsing sequence cache: {self._seq_cache}")
            return

        if self._seq_cache.exists():shutil.rmtree(self._seq_cache)
        self._seq_cache.mkdir(parents=True)
        pickle.dump(current,open(p,"wb"),protocol=pickle.HIGHEST_PROTOCOL)

    def _fit_lab_bins(self,train_subjects,admissions):
        p=self._seq_cache/"lab_bin_edges.pkl"
        if p.exists():return pickle.load(open(p,"rb"))

        tmp=self._seq_cache/"_lab_bins"
        if tmp.exists():shutil.rmtree(tmp)
        tmp.mkdir()

        for b in range(self.N_BUCKETS):
            q=self._cache/f"bucket_{b:02d}.csv.gz"
            if not q.exists():continue

            x=pd.read_csv(q,dtype={"concept_id":"string","lab_itemid":"string"})
            x=x[x.subject_id.isin(train_subjects)&x.lab_itemid.notna()].copy()
            if x.empty:continue

            x["event_time"]=self._parse_time(x.event_time)
            x["hadm_id"]=pd.to_numeric(x.hadm_id,errors="coerce").astype("Int64")

            keep=[]
            for _,g in x.groupby("subject_id",sort=False):
                g=self._attach_hadm(g.copy(),admissions)
                g=g[g.hadm_id.notna()]
                if not g.empty:keep.append(g[["lab_itemid","lab_value"]])

            if not keep:continue

            x=pd.concat(keep,ignore_index=True)
            x["_b"]=pd.to_numeric(x.lab_itemid,errors="coerce").fillna(0).astype(np.int64)%self.N_BUCKETS

            for j,g in x.groupby("_b"):
                f=tmp/f"{int(j):02d}.csv.gz"
                g[["lab_itemid","lab_value"]].to_csv(
                    f,mode="a",index=False,header=not f.exists(),
                    compression={"method":"gzip","compresslevel":1}
                )

        edges={}
        for q in tmp.glob("*.csv.gz"):
            x=pd.read_csv(q,dtype={"lab_itemid":"string"})
            for itemid,g in x.groupby("lab_itemid"):
                edges[str(itemid)]=np.quantile(
                    g.lab_value.to_numpy(float),[0.2,0.4,0.6,0.8]
                ).astype(np.float32)

        pickle.dump(edges,open(p,"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        shutil.rmtree(tmp)
        self.args.logger.write(f"\nLab bin edges fitted: {len(edges)}")
        return edges

    def _build_vocab(self,train_subjects,admissions,bins):
        tokens=set()

        for b in range(self.N_BUCKETS):
            p=self._cache/f"bucket_{b:02d}.csv.gz"
            if not p.exists():continue

            x=pd.read_csv(p,dtype={"concept_id":"string","lab_itemid":"string"})
            x=x[x.subject_id.isin(train_subjects)].copy()
            if x.empty:continue

            x["event_time"]=self._parse_time(x.event_time)
            x["hadm_id"]=pd.to_numeric(x.hadm_id,errors="coerce").astype("Int64")

            for _,g in x.groupby("subject_id",sort=False):
                g=self._attach_hadm(g.copy(),admissions)
                g=g[g.hadm_id.notna()].copy()
                if not g.empty:
                    tokens.update(self._encode_labs(g,bins).concept_id.dropna().astype(str).unique())

        vocab={t:i for i,t in enumerate(self.SPECIAL_TOKENS)}
        for t in sorted(tokens):
            if t not in vocab:vocab[t]=len(vocab)
        return vocab

    def _get_vocab(self,train_subjects,admissions,bins):
        p=self._seq_cache/"vocab.pkl"
        if p.exists():return pickle.load(open(p,"rb"))
        vocab=self._build_vocab(train_subjects,admissions,bins)
        pickle.dump(vocab,open(p,"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        self.args.logger.write(f"\nVocabulary built: {len(vocab)}")
        return vocab

    def _save_bucket(self,b,d):
        p=self._seq_cache/f"bucket_{b:02d}.pkl"
        tmp=self._seq_cache/f"bucket_{b:02d}.tmp"
        pickle.dump(d,open(tmp,"wb"),protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(p)

    def _process_bucket(self,b,ss,ages,admissions,bins,vocab):
        out=self._seq_cache/f"bucket_{b:02d}.pkl"
        if out.exists():
            self.args.logger.write(f"\nBucket {b+1}/{self.N_BUCKETS}: cached")
            return

        p=self._cache/f"bucket_{b:02d}.csv.gz"
        if not p.exists():return

        e=pd.read_csv(p,dtype={"concept_id":"string","lab_itemid":"string"})
        e["event_time"]=self._parse_time(e.event_time)
        e["hadm_id"]=pd.to_numeric(e.hadm_id,errors="coerce").astype("Int64")
        e=e.sort_values(["subject_id","event_time","hadm_id"],kind="stable")

        seqs,sids,codes=[],[],[]
        total=before=after=dropped=max_visit=0

        for sid,g in e.groupby("subject_id",sort=False):
            sid=int(sid)

            if sid in ss["train"]:code=0
            elif sid in ss["val"]:code=1
            elif sid in ss["test"]:code=2
            else:continue

            g=g.copy()
            total+=len(g)
            before+=int(g.hadm_id.isna().sum())
            g=self._attach_hadm(g,admissions)

            m=int(g.hadm_id.isna().sum())
            after+=m
            dropped+=m
            g=g[g.hadm_id.notna()].copy()

            if g.empty:continue

            x=self._make_sequence(g,sid,ages,bins,vocab)
            max_visit=max(max_visit,int(x.visit_order.max()))

            for c in self._chunk(x):
                seqs.append(c)
                sids.append(sid)
                codes.append(code)

        self._save_bucket(b,{
            "sequences":seqs,
            "subject_ids":sids,
            "split":np.asarray(codes,dtype=np.int8),
            "max_visit":max_visit,
            "total_events":total,
            "missing_before":before,
            "missing_after":after,
            "dropped":dropped
        })

        self.args.logger.write(
            f"\nBucket {b+1}/{self.N_BUCKETS}: saved, chunks={len(seqs)}, events={total}, "
            f"missing_hadm={before}->{after}, dropped={dropped}, max_visits={max_visit}"
        )

    def prepare_inputs(self):
        ss=self._get_subject_splits()
        subjects=set().union(*ss.values())

        assert len(ss["train"])>0
        assert len(ss["val"])>0
        assert ss["train"].isdisjoint(ss["val"])

        self._build_event_cache(subjects)
        self._prepare_seq_cache(ss)

        admissions=self._load_admissions(subjects)
        ages=self._load_ages(subjects)
        bins=self._fit_lab_bins(ss["train"],admissions)
        vocab=self._get_vocab(ss["train"],admissions,bins)

        for b in range(self.N_BUCKETS):
            self._process_bucket(b,ss,ages,admissions,bins,vocab)

        sequences,subject_ids=[],[]
        splits={"train":[],"val":[],"test":[]}
        total=before=after=dropped=max_visit=0

        for b in range(self.N_BUCKETS):
            p=self._seq_cache/f"bucket_{b:02d}.pkl"
            if not p.exists():continue

            d=pickle.load(open(p,"rb"))
            off=len(sequences)
            sequences.extend(d["sequences"])
            subject_ids.extend(d["subject_ids"])

            max_visit=max(max_visit,d["max_visit"])
            total+=d["total_events"]
            before+=d["missing_before"]
            after+=d["missing_after"]
            dropped+=d["dropped"]

            for code,name in enumerate(["train","val","test"]):
                splits[name].extend((np.where(d["split"]==code)[0]+off).tolist())

        splits={k:np.asarray(v,dtype=np.int64) for k,v in splits.items()}
        lengths=np.asarray([len(x) for x in sequences],dtype=np.int32)

        assert len(sequences)>0
        assert len(splits["train"])>0
        assert len(splits["val"])>0
        assert lengths.max()<=int(self.args.max_seq_len)
        assert len(vocab)>len(self.SPECIAL_TOKENS)

        self.args.V=self.args.vocab_size=len(vocab)
        self.args.N=len(sequences)
        self.args.max_num_visits=max(int(getattr(self.args,"max_num_visits",1024)),max_visit)
        self.var_to_ind=vocab

        self.input_dict={
            "ehrmamba_sequences":sequences,
            "subject_ids":np.asarray(subject_ids,dtype=np.int64),
            "vocab":vocab,
            "vocab_size":len(vocab),
            "lab_bin_edges":bins,
            "max_seq_len":int(self.args.max_seq_len),
            "max_num_visits":self.args.max_num_visits,
            "splits":splits,
            "var_to_ind":vocab
        }

        pct=100.*dropped/total if total else 0.

        self.args.logger.write(
            f"\nEHRMamba-Paper PRETRAIN: patients={len(set(subject_ids))}, chunks={len(sequences)}, "
            f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}, "
            f"vocab={len(vocab)}, median_len={int(np.median(lengths))}, max_len={int(lengths.max())}, "
            f"max_visits={max_visit}, events={total}, missing_hadm={before}->{after}, "
            f"dropped={dropped} ({pct:.2f}%)"
        )

    def save_inputs(self):
        meta={k:self.input_dict[k] for k in [
            "vocab","vocab_size","lab_bin_edges","max_seq_len","max_num_visits"
        ]}
        pickle.dump(
            meta,
            open(Path(self.args.paths["output_path"])/"pt_saved_variables.pkl","wb"),
            protocol=pickle.HIGHEST_PROTOCOL
        )

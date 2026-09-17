import numpy as np
import torch
from ts_model_training.cycler import CycleIndex, CycleIndexBalanced

class Batcher:
    def __init__(self, args, input_dict):
        self.args = args
        self.input_dict = input_dict
        self.splits = self.input_dict.get("splits")
        self.demo = self.input_dict.get("demo_norm")
        self.y = self.input_dict.get("target") # none if attribute missing
        self.set_cycler()

    def _get_indices(self, ind):
        if ind is None:
            return self.train_cycler.get_batch_ind()
        return ind

    def get_batch(self, ind=None):
        raise NotImplementedError
    
    def set_cycler(self):    
        if self.args.train_mode == "pretrain":
            self.train_cycler = CycleIndex(self.splits['train'], self.args.train_batch_size)
        elif self.args.stratify_batch > 0:
            self.train_cycler = CycleIndexBalanced(self.splits['train'], self.y[self.splits['train']], self.args.train_batch_size, self.args.stratify_batch)
        else:
            self.train_cycler = CycleIndex(self.splits['train'], self.args.train_batch_size)

class BatcherA(Batcher):
    def __init__(self, args, input_dict):
        super().__init__(args, input_dict) 
        self.X = self.input_dict.get("X")

    def get_batch(self, ind=None):
        ind = self._get_indices(ind)
        return {
            'ts': torch.FloatTensor(self.X[ind]),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }

class BatcherB(Batcher):
    def __init__(self, args, input_dict):
        super().__init__(args, input_dict) 
        self.values = self.input_dict.get("values")
        self.mask = self.input_dict.get("mask")
        if self.args.model_type == "grud":
            self.deltas = self.input_dict.get("deltas")
        elif self.args.model_type == "interpnet":
            self.times = self.input_dict.get("times")
            self.holdout_masks = self.input_dict.get("holdout_masks")
        else:
            raise ValueError(f"Unsupported model type: {self.args.model_type}")  

    def _pad_and_stack(self, sequences, pad_mats):
        return torch.FloatTensor(
            np.stack([
                np.concatenate((seq, pad), axis=0) if len(seq) > 0 else pad
                for seq, pad in zip(sequences, pad_mats)
            ])
        )

    def _make_pad_mats(self, num_timestamps, V):
        max_timestamps = max(num_timestamps)
        pad_lens = max_timestamps - num_timestamps
        pad_mats = [np.zeros((l, V)) for l in pad_lens]
        return pad_mats, pad_lens

    def get_batch(self, ind=None):
        ind = self._get_indices(ind)
        if self.args.model_type == "grud":
            return self.get_batch_grud(ind)
        elif self.args.model_type == "interpnet":
            return self.get_batch_interpnet(ind)
        else:
            raise ValueError(f"Unsupported model type: {self.args.model_type}")  

    def get_batch_grud(self, ind=None):
        deltas = [self.deltas[i] for i in ind]
        values = [self.values[i] for i in ind]
        masks = [self.mask[i] for i in ind]

        num_timestamps = np.array([len(d) for d in deltas])
        pad_mats, _ = self._make_pad_mats(num_timestamps, self.args.V)

        return {
            'x_t': self._pad_and_stack(values, pad_mats),
            'm_t': self._pad_and_stack(masks, pad_mats),
            'delta_t': self._pad_and_stack(deltas, pad_mats),
            'seq_len': torch.LongTensor(num_timestamps),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }

    def get_batch_interpnet(self, ind=None):
        times = [self.times[i] for i in ind]
        values = [self.values[i] for i in ind]
        masks = [self.mask[i] for i in ind]
        hmasks = [self.holdout_masks[i] for i in ind]

        num_timestamps = np.array(list(map(len, times)))
        pad_mats, pad_lens = self._make_pad_mats(num_timestamps, self.args.V)

        return {
            't': torch.FloatTensor([t + [0] * p for t, p in zip(times, pad_lens)]),
            'x': self._pad_and_stack(values, pad_mats),
            'm': self._pad_and_stack(masks, pad_mats),
            'h': self._pad_and_stack(hmasks, pad_mats),
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }

class BatcherC_unsup(Batcher):
    def __init__(self, args, input_dict):
        super().__init__(args, input_dict) 
        self.values = self.input_dict.get("values")
        self.times = self.input_dict.get("times")
        self.varis = self.input_dict.get("varis")
        self.timestamps = self.input_dict.get("timestamps")     
        self.max_minute = self.args.window_forecast #7*24*60 #args.window_forecast*24*60 if not None else 7*24*60
        self.pred_int = self.args.window_pred #1*24*60 #12*60 1 day forecasting window as opposed to 12 hours

    def get_batch(self, ind=None):
        ind = self._get_indices(ind)
        
        input_values = []
        input_times = []
        input_varis = []
        forecast_values = torch.zeros((len(ind),self.args.V))
        forecast_mask = torch.zeros((len(ind),self.args.V), dtype=torch.int)
        for b,i in enumerate(ind):
            t1 = np.random.choice(self.timestamps[i]) # minutes
            curr_times = self.times[i]
            for ix in range(len(curr_times)-1,-1,-1):
                if curr_times[ix]==t1:
                    t1_ix = ix+1 # start of prediction window
                    break
            t0_ix = max(0,t1_ix-self.args.max_obs)

            while curr_times[t0_ix]<t1-self.max_minute:
                t0_ix += 1
            if t1>self.max_minute: # shift times
                diff = t1-self.max_minute
                input_times.append(list(np.array(self.times[i][t0_ix:t1_ix])-diff))
            else:
                input_times.append(self.times[i][t0_ix:t1_ix])
            input_values.append(self.values[i][t0_ix:t1_ix])
            input_varis.append(self.varis[i][t0_ix:t1_ix])

            t2 = t1+self.pred_int
            for t2_ix in range(t1_ix, len(curr_times)):
                if curr_times[t2_ix]>t2:
                    break
            # t2_ix: last+1 for prediction window
            curr_varis = self.varis[i]
            curr_values = self.values[i]
            for ix in range(t2_ix-1,t1_ix-1,-1):
                vari = curr_varis[ix]
                val = curr_values[ix]
                forecast_mask[b,vari] = 1
                forecast_values[b,vari] = val

        num_obs = list(map(len, input_values))
        max_obs = max(num_obs)
        pad_lens = max_obs-np.array(num_obs)
        values = [x+[0]*(l) for x,l in zip(input_values,pad_lens)]
        times = [x+[0]*(l) for x,l in zip(input_times,pad_lens)]
        varis = [x+[0]*(l) for x,l in zip(input_varis,pad_lens)]
        values, times = torch.FloatTensor(values), torch.FloatTensor(times)
        times = times/self.max_minute*2-1
        varis = torch.IntTensor(varis)
        obs_mask = [[1]*l1+[0]*l2 for l1,l2 in zip(num_obs,pad_lens)]
        obs_mask = torch.IntTensor(obs_mask)

        #self.args.logger.write(num_obs)

        return {'values':values, 'times':times, 'varis':varis,
                'obs_mask':obs_mask, 
                'demo':torch.FloatTensor(self.demo[ind]),
                'forecast_values':forecast_values,
                'forecast_mask':forecast_mask}


class BatcherC_sup(Batcher):
    def __init__(self, args, input_dict):
        super().__init__(args, input_dict) 
        self.values = self.input_dict.get("values")
        self.times = self.input_dict.get("times")
        self.varis = self.input_dict.get("varis")

    def get_batch(self, ind=None):
        ind = self._get_indices(ind)

        num_obs = [len(self.values[i]) for i in ind]
        max_obs = max(num_obs)
        pad_lens = max_obs - np.array(num_obs)

        values = [self.values[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        times = [self.times[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        varis = [self.varis[i]+[0]*(l) for i,l in zip(ind,pad_lens)]
        values, times = torch.FloatTensor(values), torch.FloatTensor(times)
        varis = torch.IntTensor(varis)
        obs_mask = [[1]*l1+[0]*l2 for l1,l2 in zip(num_obs,pad_lens)]
        obs_mask = torch.IntTensor(obs_mask)

        return {
            'values': values,
            'times': times,
            'varis': varis,
            'obs_mask': obs_mask,
            'demo': torch.FloatTensor(self.demo[ind]),
            'labels': torch.FloatTensor(self.y[ind])
        }


class BatcherEHRMamba(Batcher):
    def __init__(self, args, input_dict):
        super().__init__(args, input_dict)
        self.event_ids = self.input_dict["event_ids"]
        self.values = self.input_dict["values"]
        self.event_mask = self.input_dict["event_mask"]
        self.event_times = self.input_dict["event_times"]
        self.lengths = self.input_dict["lengths"]

    def get_batch(self, ind=None):
        ind = self._get_indices(ind)
        return {
            "event_ids": torch.LongTensor(self.event_ids[ind]),
            "values": torch.FloatTensor(self.values[ind]),
            "event_mask": torch.FloatTensor(self.event_mask[ind]),
            "event_times": torch.FloatTensor(self.event_times[ind]),
            "lengths": torch.LongTensor(self.lengths[ind]),
            "demo": torch.FloatTensor(self.demo[ind]),
            "labels": torch.FloatTensor(self.y[ind]),
        }

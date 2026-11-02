import torch
import torch.nn as nn

from .ts_model import TimeSeriesModel
from .ts_mamba import _MambaResidualBlock

PAD_ID = 0
CLS_ID = 1
REG_ID = 2
UNK_ID = 3
LAB_ID_OFFSET = 4


class Time2Vec(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        if d_model < 2:
            raise ValueError(f"d_model must be >= 2 for Time2Vec, got {d_model}")

        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d_model - 1)

    def forward(self, t):
        t = t.unsqueeze(-1)
        linear_part = self.linear(t)
        periodic_part = torch.sin(self.periodic(t))
        return torch.cat([linear_part, periodic_part], dim=-1)


class EHRLabEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout, value_encoding):
        super().__init__()
        self.value_encoding = value_encoding

        self.concept_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        if value_encoding == "continuous":
            self.value_encoder = nn.Linear(1, d_model)
        elif value_encoding == "bins5":
            self.value_encoder = nn.Embedding(5, d_model)
        else:
            raise ValueError(f"Unknown value_encoding: {value_encoding}")

        self.time_emb = Time2Vec(d_model)
        self.fusion_proj = nn.Linear(3 * d_model, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout)

    def forward(self, event_ids, values, event_mask, event_times):
        concept = self.concept_emb(event_ids)
        measurement_mask = event_mask.bool().unsqueeze(-1)

        if self.value_encoding == "continuous":
            value = self.value_encoder(values.unsqueeze(-1).float())
        else:
            value = self.value_encoder(values.long())
        value = value.masked_fill(~measurement_mask, 0.0)

        time = self.time_emb(event_times)
        time = time.masked_fill(~measurement_mask, 0.0)

        x = torch.cat([concept, value, time], dim=-1)
        x = self.fusion_proj(x)
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)

        nonpad_mask = (event_ids != PAD_ID).unsqueeze(-1)
        x = x.masked_fill(~nonpad_mask, 0.0)

        return x


class EHRMAMBA_LAB_TS(TimeSeriesModel):
    def __init__(self, args):
        super().__init__(args)
        value_encoding = str(getattr(args, "value_encoding", "continuous"))
        self.pretrain = args.train_mode == "pretrain"
        self.pretrain_protocol = getattr(args, "pretrain_protocol", "ehrmamba_adaptation")
        self.V = int(args.V)
        self.variable_mask_prob = float(args.dropout)

        H = int(args.hid_dim)
        dropout = float(args.dropout)

        if self.pretrain and self.pretrain_protocol == "strats_matched":
            vocab_size = self.V + 4
        else:
            vocab_size = int(args.vocab_size)

        self.embedding = EHRLabEmbedding(vocab_size=vocab_size, d_model=H, dropout=dropout,value_encoding=value_encoding)

        self.layers = nn.ModuleList([
            _MambaResidualBlock(
                H,
                int(getattr(args, "d_state", 16)),
                int(getattr(args, "d_conv", 4)),
                int(getattr(args, "expand", 2)),
                dropout,
            )
            for _ in range(int(args.num_layers))
        ])

        self.final_norm = nn.LayerNorm(H)
        self.dropout = nn.Dropout(dropout)

        if self.pretrain:
            if self.pretrain_protocol == "strats_matched":
                self.forecast_head = nn.Linear(self.ts_demo_emb_size, int(args.V))
            else:
                self.forecast_head = nn.Linear(H, int(args.V))

    def _apply_strats_variable_mask(self, event_ids, values, event_mask, event_times, lengths):
        B = event_ids.size(0)
        var_mask = torch.rand((B, self.V), device=event_ids.device) <= self.variable_mask_prob

        out_ids = torch.zeros_like(event_ids)
        out_values = torch.zeros_like(values)
        out_mask = torch.zeros_like(event_mask)
        out_times = torch.zeros_like(event_times)
        out_lengths = torch.zeros_like(lengths)

        out_ids[:, 0] = CLS_ID

        for b in range(B):
            L = int(lengths[b].item())
            ids = event_ids[b, 1:L-1]
            keep = ~var_mask[b, (ids - LAB_ID_OFFSET).long()]
            n = int(keep.sum().item())

            out_ids[b, 1:1+n] = ids[keep]
            out_values[b, 1:1+n] = values[b, 1:L-1][keep]
            out_mask[b, 1:1+n] = event_mask[b, 1:L-1][keep]
            out_times[b, 1:1+n] = event_times[b, 1:L-1][keep]
            out_ids[b, n+1] = REG_ID
            out_lengths[b] = n + 2

        return out_ids, out_values, out_mask, out_times, out_lengths

    def forecast_loss(self, ts_emb, forecast_values, forecast_mask):
        pred = self.forecast_head(ts_emb)
        mask = forecast_mask.float()
        denom = mask.sum()

        if denom.item() == 0:
            raise RuntimeError("EHRMamba pretrain batch has no forecast targets")

        return (mask * (pred - forecast_values) ** 2).sum() / denom

    def forward(
        self,
        event_ids,
        values,
        event_mask,
        event_times,
        lengths,
        demo,
        labels=None,
        forecast_values=None,
        forecast_mask=None,
    ):
        if self.training and self.pretrain and self.pretrain_protocol == "strats_matched":
            event_ids, values, event_mask, event_times, lengths = self._apply_strats_variable_mask(
                event_ids, values, event_mask, event_times, lengths
            )

        x = self.embedding(event_ids, values, event_mask, event_times)

        h = x
        for layer in self.layers:
            h = layer(h)

        C = self.final_norm(h)

        idx = lengths - 1
        batch_ix = torch.arange(C.size(0), device=C.device)
        ts_emb = C[batch_ix, idx]
        ts_emb = self.dropout(ts_emb)

        if self.pretrain:
            if self.pretrain_protocol == "strats_matched":
                demo_emb = self.demo_emb(demo)
                pretrain_emb = torch.cat((ts_emb, demo_emb), dim=-1)
            else:
                pretrain_emb = ts_emb
            loss = self.forecast_loss(pretrain_emb, forecast_values, forecast_mask)
            return loss, pretrain_emb

        demo_emb = self.demo_emb(demo)
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)

        logits = self.binary_head(ts_demo_emb)[:, 0]

        return logits, ts_demo_emb

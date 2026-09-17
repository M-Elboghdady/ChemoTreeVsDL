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
    def __init__(self, vocab_size, d_model, dropout):
        super().__init__()

        self.concept_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.value_proj = nn.Linear(1, d_model)
        self.time_emb = Time2Vec(d_model)
        self.fusion_proj = nn.Linear(3 * d_model, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout)

    def forward(self, event_ids, values, event_mask, event_times):
        concept = self.concept_emb(event_ids)
        measurement_mask = event_mask.bool().unsqueeze(-1)

        value = self.value_proj(values.unsqueeze(-1))
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

        H = int(args.hid_dim)
        dropout = float(args.dropout)
        vocab_size = int(args.vocab_size)

        self.embedding = EHRLabEmbedding(vocab_size=vocab_size, d_model=H, dropout=dropout)

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

    def forward(self, event_ids, values, event_mask, event_times, lengths, demo, labels=None):
        x = self.embedding(event_ids, values, event_mask, event_times)

        h = x
        for layer in self.layers:
            h = layer(h)

        C = self.final_norm(h)

        idx = lengths - 1
        batch_ix = torch.arange(C.size(0), device=C.device)
        ts_emb = C[batch_ix, idx]
        ts_emb = self.dropout(ts_emb)

        demo_emb = self.demo_emb(demo)
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)

        logits = self.binary_head(ts_demo_emb)[:, 0]

        return logits, ts_demo_emb

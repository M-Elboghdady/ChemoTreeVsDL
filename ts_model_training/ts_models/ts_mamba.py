from .ts_model import TimeSeriesModel
import torch.nn as nn
import torch
from mamba_ssm import Mamba

# MODEL PARAMS:
# hid_dim
# num_layers
# dropout
# d_state
# d_conv
# expand

class _MambaResidualBlock(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        h = self.norm(x)
        h = self.mamba(h)
        h = self.dropout(h)
        return residual + h


class MAMBA_TS(TimeSeriesModel):
    def __init__(self, args):
        super().__init__(args)

        input_dim = args.V * len(args.variant)
        d_model = int(args.hid_dim)
        num_layers = int(args.num_layers)
        dropout = float(args.dropout)
        d_state = int(getattr(args, "d_state", 16))
        d_conv = int(getattr(args, "d_conv", 4))
        expand = int(getattr(args, "expand", 2))

        self.input_proj = nn.Linear(int(input_dim), d_model)
        self.layers = nn.ModuleList(
            [
                _MambaResidualBlock(d_model, d_state, d_conv, expand, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ts, demo, labels=None):

        h = self.input_proj(ts)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)

        ts_emb = h[:, -1, :]
        ts_emb = self.dropout(ts_emb)

        demo_emb = self.demo_emb(demo)
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)

        logits = self.binary_head(ts_demo_emb)[:, 0]
        return logits, ts_demo_emb

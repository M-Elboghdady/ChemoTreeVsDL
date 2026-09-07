import torch
import torch.nn as nn
from .ts_mamba import MAMBA_TS

class MAMBA_FUSION_TS(MAMBA_TS):
    def __init__(self, args):
        super().__init__(args)
        H = int(args.hid_dim)
        d_a = int(getattr(args, "d_a", 32))
        self.attn_Wa = nn.Linear(H, d_a)
        self.attn_ua = nn.Linear(d_a, 1, bias=False)

    def forward(self, ts, demo, labels=None):
        # Same encoder as plain Mamba → C_E [B, L, H]
        h = self.input_proj(ts)
        for layer in self.layers:
            h = layer(h)
        C = self.final_norm(h)   # [B, L, H]
        # Fusion self-attention scores over time
        a = self.attn_ua(torch.tanh(self.attn_Wa(C))).squeeze(-1)  # [B, L]
        alpha = torch.softmax(a, dim=1)
        z_T = torch.sum(alpha.unsqueeze(-1) * C, dim=1)  # [B, H]
        z_T = self.dropout(z_T)
        # Same static pathway + classifier as plain Mamba
        demo_emb = self.demo_emb(demo)
        ts_demo_emb = torch.cat((z_T, demo_emb), dim=-1)
        logits = self.binary_head(ts_demo_emb)[:, 0]
        return logits, ts_demo_emb

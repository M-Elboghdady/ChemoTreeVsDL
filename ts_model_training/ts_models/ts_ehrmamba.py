import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
from .ts_model import TimeSeriesModel


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Time2Vec(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, d-1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        return torch.cat([self.linear(x), torch.sin(self.periodic(x))], -1)

class EHRMambaEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model,time_size=32,type_vocab_size=9,max_num_visits=1024):
        super().__init__()
        self.concept_emb=nn.Embedding(vocab_size,d_model,padding_idx=0)
        self.type_emb=nn.Embedding(type_vocab_size,d_model,padding_idx=0)
        self.segment_emb=nn.Embedding(3,d_model,padding_idx=0)
        self.visit_emb=nn.Embedding(max_num_visits+1,d_model,padding_idx=0)
        self.age_emb=Time2Vec(time_size)
        self.time_emb=Time2Vec(time_size)
        self.temporal=nn.Linear(d_model+2*time_size,d_model)

    def forward(self,token_ids,type_ids,ages,times,visit_orders,visit_segments):
        c=self.concept_emb(token_ids)
        m=(visit_orders>0).unsqueeze(-1)
        a=self.age_emb(ages).masked_fill(~m,0.)
        t=self.time_emb(times).masked_fill(~m,0.)
        x=self.temporal(torch.cat([c,a,t],-1))
        x=x+self.type_emb(type_ids)+self.segment_emb(visit_segments)+self.visit_emb(visit_orders)
        return x.masked_fill((token_ids==0).unsqueeze(-1),0.)

class EHRMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class EHRMambaBackbone(nn.Module):
    def __init__(self, d_model, num_layers, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EHRMambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        for layer in self.layers: x = layer(x)
        return self.norm(x)


class EHRMAMBA_PAPER_TS(TimeSeriesModel):
    def __init__(self, args):
        super().__init__(args)
        self.pretrain = args.train_mode == "pretrain"

        H, V = int(args.hid_dim), int(args.vocab_size)
        dropout = float(args.dropout)

        self.embedding = EHRMambaEmbedding(
            V, H,
            int(getattr(args, "time_size", 32)),
            int(getattr(args, "type_vocab_size", 9)),
            int(getattr(args, "max_num_visits", 1024))
        )

        self.backbone = EHRMambaBackbone(
            H, int(args.num_layers),
            int(getattr(args, "d_state", 16)),
            int(getattr(args, "d_conv", 4)),
            int(getattr(args, "expand", 2)),
            dropout
        )

        self.dropout = nn.Dropout(dropout)

        if self.pretrain:
            if hasattr(self, "binary_head"): del self.binary_head
            if hasattr(self, "demo_emb"): del self.demo_emb
            self.forecast_norm = RMSNorm(H)
            self.forecast_head = nn.Linear(H, V, bias=False)
        else:
            self.binary_head = nn.Linear(H, 1)

    def forward(self, token_ids, type_ids, ages, times, visit_orders, visit_segments,
                lengths=None, demo=None, attention_mask=None, labels=None, forecast_mask=None):

        h = self.backbone(self.embedding(
            token_ids, type_ids, ages, times, visit_orders, visit_segments
        ))

        if self.pretrain:
            logits = self.forecast_head(self.forecast_norm(h))
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100
            )
            return loss, h

        b = torch.arange(h.size(0), device=h.device)
        emb = self.dropout(h[b, lengths-1])
        return self.binary_head(emb)[:, 0], emb

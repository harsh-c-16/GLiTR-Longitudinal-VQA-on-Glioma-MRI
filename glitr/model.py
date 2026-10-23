"""GLiTR: learned per-component weights over the 4 sequence features, one token per scan (features, change since the previous
scan, question, scan time), a small transformer over the series, and a cascade head: Stable vs. Change, then
Decrease / Increase / Fluctuating.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MOD, FEAT = 4, 768
Q_DIM, PROJ, HID, TIME_PE = 256, 256, 128, 32
MAX_WEEKS = 52.0
BERT = "dmis-lab/biobert-v1.1"
COMP_ID = {"necrosis": 0, "edema": 1, "tumor": 2}


def bert_path():
    """Local snapshot of BioBERT if it is cached (no network access), else the hub id (downloads it)."""
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(BERT, local_files_only=True)
    except Exception:
        return BERT
    has_weights = any(os.path.exists(os.path.join(path, f)) for f in ("pytorch_model.bin", "model.safetensors"))
    return path if has_weights else BERT


class SinTime(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("freqs", torch.exp(torch.arange(0, TIME_PE, 2).float() * -(math.log(10000.0) / TIME_PE)))

    def forward(self, t):
        a = (t / MAX_WEEKS).unsqueeze(-1) * self.freqs
        pe = torch.zeros(*t.shape, TIME_PE, device=t.device)
        pe[..., 0::2], pe[..., 1::2] = torch.sin(a), torch.cos(a)
        return pe


class Model(nn.Module):
    def __init__(self, dropout=0.27, bert=None, feat=FEAT):
        super().__init__()
        self.vis = VIS = N_MOD * feat
        self.comp = nn.Embedding(3, N_MOD)
        with torch.no_grad():                                # initial sequence weights per component
            self.comp.weight.copy_(torch.tensor([[0.8, 1.2, 0.3, 0.4], [0.2, 0.4, 1.1, 1.5], [1.4, 0.7, 0.6, 1.0]]))
        if bert is None:
            from transformers import AutoModel
            bert = AutoModel.from_pretrained(bert_path())
        self.bert = bert.requires_grad_(False)
        self.q_proj = nn.Linear(768, Q_DIM)
        self.vis_proj = nn.Sequential(nn.Linear(VIS, PROJ), nn.LayerNorm(PROJ), nn.GELU())
        self.time_enc = SinTime()
        self.step_mlp = nn.Sequential(nn.Linear(VIS + PROJ + Q_DIM + TIME_PE, 256), nn.LayerNorm(256), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(256, HID), nn.LayerNorm(HID), nn.GELU(),
                                      nn.Dropout(dropout))
        self.time_proj = nn.Linear(TIME_PE, HID)
        self.mixer = nn.TransformerEncoder(nn.TransformerEncoderLayer(HID, 4, 2 * HID, 0.1, batch_first=True,
                                                                      activation="gelu"), 2)
        self.attn_pool = nn.Linear(HID, 1)
        self.binary_head = nn.Sequential(nn.Linear(2 * HID, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 2))
        self.triclass_head = nn.Sequential(nn.Linear(4 * HID, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 3))

    def forward(self, feats, time, tmask, comp, q_ids, q_mask):
        """feats [B,T,4,feat], time/tmask [B,T]. Returns (binary_logits [B,2], triclass_logits [B,3])."""
        b, t = feats.shape[:2]
        w = F.softmax(self.comp(comp), dim=-1)
        x = (feats * w.view(b, 1, N_MOD, 1)).reshape(b, t, self.vis)
        tmask = tmask.to(x.dtype)
        n = tmask.sum(1).clamp(min=1.0)
        last = (n.long() - 1).clamp(0, t - 1)
        bi = torch.arange(b, device=x.device)

        proj = self.vis_proj(x)
        deltas = torch.zeros_like(proj)
        deltas[:, 1:] = proj[:, 1:] - proj[:, :-1]
        q = self.q_proj(self.bert(input_ids=q_ids, attention_mask=q_mask).last_hidden_state[:, 0])
        pe = self.time_enc(time)
        h = self.step_mlp(torch.cat([x, deltas, q.unsqueeze(1).expand(b, t, Q_DIM), pe], dim=2))
        mixed = h + self.mixer(h + self.time_proj(pe), src_key_padding_mask=tmask <= 0)
        m = tmask.unsqueeze(-1)
        mixed = mixed * m

        pair = (tmask[:, 1:] * tmask[:, :-1]).unsqueeze(-1)
        path = (mixed[:, 1:] - mixed[:, :-1]).abs().mul(pair).sum(1)
        diff = mixed[bi, last] - mixed[:, 0]
        mono = diff.abs() / (path + 1e-6)
        mean = mixed.sum(1) / n.unsqueeze(-1)
        std = (((mixed - mean.unsqueeze(1)) * m).pow(2).sum(1) / n.unsqueeze(-1) + 1e-6).sqrt()
        att = self.attn_pool(mixed).squeeze(-1).masked_fill(tmask <= 0, float("-inf")).softmax(dim=1)
        pooled = (mixed * att.unsqueeze(-1)).sum(1)
        return self.binary_head(torch.cat([mean, std], 1)), self.triclass_head(torch.cat([pooled, diff, mean, mono], 1))


def decode(binary_logits, triclass_logits):
    """4-class prediction (0 Decrease, 1 Stable, 2 Increase, 3 Fluctuating) from the cascade outputs."""
    tri = torch.tensor([0, 2, 3], device=triclass_logits.device)[triclass_logits.argmax(1)]
    return torch.where(binary_logits.argmax(1) == 0, torch.ones_like(tri), tri)


def class_probs(binary_logits, triclass_logits):
    """4-class scores that sum to 1, for AUROC: P(Stable) = gate, P(direction) = P(Change) x direction softmax."""
    pb, pt = F.softmax(binary_logits.float(), -1), F.softmax(triclass_logits.float(), -1)
    ch = pb[:, 1:2]
    return torch.cat([ch * pt[:, 0:1], pb[:, 0:1], ch * pt[:, 1:2], ch * pt[:, 2:3]], 1)


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, reduction="none")
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()

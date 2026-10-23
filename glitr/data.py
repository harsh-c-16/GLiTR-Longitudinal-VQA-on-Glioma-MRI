"""Samples, folds and batching over precomputed features (see preprocess.py)."""
import os
import zlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

from .model import COMP_ID, N_MOD

QTEXT = {1: "What was the overall trend of the necrosis volume?",
         2: "What was the overall trend of the Contrast-enhancing tumor volume?",
         3: "What was the overall trend of the Edema volume?",
         4: "What was the net change in the necrosis volume?",
         5: "What was the net change in the Contrast-enhancing tumor volume?",
         6: "What was the net change in the Edema volume?"}
Q_LEN = 32


def cls_of(direction):
    """Direction (-1, 0, 1, 2) -> class index (0 Decrease, 1 Stable, 2 Increase, 3 Fluctuating)."""
    return 3 if direction == 2 else direction + 1


def load_gt(path):
    gt = defaultdict(list)
    for r in pd.read_csv(path).itertuples():
        gt[r.Patient_ID].append((r.Component.lower(), int(r.Question_ID), int(r.Trend_Direction)))
    return gt


def load_features(feat_dir, patients):
    return {p: torch.load(os.path.join(feat_dir, f"{p}.pt"), weights_only=True) for p in patients}


def build_folds(gt, patients, n_splits=5, seed=42):
    """Patient-grouped, stratified on each patient's majority direction."""
    pats = np.array(sorted(p for p in patients if p in gt))
    major = np.array([Counter(d for _, _, d in gt[p]).most_common(1)[0][0] for p in pats])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(pats[a].tolist(), pats[b].tolist()) for a, b in sgkf.split(pats, major, np.arange(len(pats)))]


def folds_from_file(path, patients):
    """Explicit outer folds from a CSV (Patient_ID, fold). Patients missing from the file are never scored and are
    always part of the training set."""
    f = pd.read_csv(path).drop_duplicates("Patient_ID").set_index("Patient_ID")["fold"]
    pats = sorted(p for p in patients)
    return [([p for p in pats if f.get(p) != k], [p for p in pats if f.get(p) == k]) for k in sorted(f.unique())]


def inner_split(train_patients, frac, seed):
    tp = np.array(sorted(train_patients))
    np.random.default_rng(seed).shuffle(tp)
    k = max(1, int(round(len(tp) * frac)))
    return sorted(tp[k:].tolist()), sorted(tp[:k].tolist())


def inner_folds(train_patients, k, seed):
    tp = np.array(sorted(train_patients))
    np.random.default_rng(seed).shuffle(tp)
    parts = np.array_split(tp, k)
    return [(sorted(np.concatenate(parts[:i] + parts[i + 1:]).tolist()), sorted(parts[i].tolist())) for i in range(k)]


def tokenize(tokenizer):
    t = {q: tokenizer(s, return_tensors="pt", padding="max_length", max_length=Q_LEN, truncation=True)
         for q, s in QTEXT.items()}
    return {q: (v["input_ids"][0], v["attention_mask"][0]) for q, v in t.items()}


class Trajectories(torch.utils.data.Dataset):
    """One item per (patient, component, question): the patient's whole series of scan features and times.
    ablation (the input ablations): "no_images" zeroes every image feature (question, scan count and times remain);
    "no_order" gives the scans in a fixed random order per patient and zeroes the times, which would give the order back."""

    def __init__(self, feats, gt, patients, tokens, ablation="none"):
        assert ablation in ("none", "no_images", "no_order")
        self.f, self.tok, self.ablation = feats, tokens, ablation
        self.s = [(p, c, q, d) for p in patients for c, q, d in gt[p]]
        self.by_class = defaultdict(list)
        for i, (_, _, _, d) in enumerate(self.s):
            self.by_class[cls_of(d)].append(i)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        pid, comp, qid, d = self.s[i]
        r = self.f[pid]
        ids, mask = self.tok[qid]
        x, t = r["feats"], r["time"]
        if self.ablation == "no_images":
            x = torch.zeros_like(x)
        elif self.ablation == "no_order":
            perm = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(zlib.crc32(pid.encode())))
            x, t = x[perm], torch.zeros_like(t)
        change = int(d != 0)
        return dict(feats=x, time=t, comp=COMP_ID[comp], q_ids=ids, q_mask=mask, cls=cls_of(d), binary=change,
                    tri=({-1: 0, 1: 1, 2: 2}[d] if change else -1))


def collate(items):
    t = max(it["feats"].shape[0] for it in items)
    feats = torch.zeros(len(items), t, N_MOD, items[0]["feats"].shape[-1])
    time = torch.zeros(len(items), t)
    tmask = torch.zeros(len(items), t)
    for i, it in enumerate(items):
        k = it["feats"].shape[0]
        feats[i, :k], time[i, :k], tmask[i, :k] = it["feats"], it["time"], 1
    stack = lambda k: torch.stack([it[k] for it in items])
    num = lambda k: torch.tensor([it[k] for it in items])
    return dict(feats=feats, time=time, tmask=tmask, comp=num("comp"), q_ids=stack("q_ids"), q_mask=stack("q_mask"),
                cls=num("cls"), binary=num("binary"), tri=num("tri"))


class BalancedBatches(torch.utils.data.Sampler):
    """One sample per class in every batch of four; an epoch has as many batches as the rarest class has samples."""

    def __init__(self, by_class, seed):
        self.by_class = {c: v for c, v in by_class.items() if v}
        self.n = min(len(v) for v in self.by_class.values())
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __iter__(self):
        order = {c: self.rng.permutation(v) for c, v in self.by_class.items()}
        for k in range(self.n):
            batch = [int(order[c][k % len(order[c])]) for c in self.by_class]
            self.rng.shuffle(batch)
            yield batch

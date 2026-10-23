"""Raw MRI -> frozen per-scan features: one .pt per patient with feats [T,4,D] and time [T] in weeks
(LUMIERE: the dataset's weeks; MU: since the first scan).

Per scan: 1 mm RAS, 128x192x192 crop at the brain center (no segmentation is read), 0.5-99.5 percentile clip, z-score.
Encoder: frozen BrainIAC ViT-B on a 96^3 resize, CLS token, 768-d.
A scan is used only if its four sequences and its segmentation file exist (the labels come from the segmentation).
"""
import argparse
import csv
import glob
import os
import re
import zlib

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import find_objects, label, zoom

SHAPE, RESIZE = (128, 192, 192), (96, 96, 96)
LUMIERE_SEQ = {"CT1": "CT1_r2s_bet_reg.nii.gz", "T1": "T1_r2s_bet_reg.nii.gz", "T2": "T2_r2s_bet_reg.nii.gz",
               "FLAIR": "FLAIR_r2s_bet_reg.nii.gz"}
MU_SEQ = {"CT1": "t1c", "T1": "t1n", "T2": "t2w", "FLAIR": "t2f"}
MU_FALLBACK_WEEKS = 11.0                                     # median MU inter-scan gap, only if days are missing


def load_ras(path):
    img = nib.as_closest_canonical(nib.load(path))
    z = img.header.get_zooms()[:3]
    return np.transpose(img.get_fdata().astype(np.float32), (2, 1, 0)), (float(z[2]), float(z[1]), float(z[0]))


def resample(vol, spacing, order):
    return zoom(vol, tuple(spacing), order=order)           # target spacing is 1 mm, so zoom == source spacing


def crop(vol, centre):
    out = np.zeros(SHAPE, dtype=vol.dtype)
    start = [c - s // 2 for c, s in zip(centre, SHAPE)]
    lo = [max(a, 0) for a in start]
    hi = [min(a + s, d) for a, s, d in zip(start, SHAPE, vol.shape)]
    out[tuple(slice(l - a, h - a) for l, h, a in zip(lo, hi, start))] = vol[tuple(slice(l, h) for l, h in zip(lo, hi))]
    return out


def brain_centre(path):
    """Center of the skull-stripped brain: the bounding box of the largest foreground blob of the T1c scan."""
    vol, sp = load_ras(path)
    brain = resample((vol > 0).astype(np.uint8), sp, 0)
    if not brain.any():
        return tuple(s // 2 for s in brain.shape)
    cc, n = label(brain)
    biggest = 1 + int(np.argmax([(cc == i).sum() for i in range(1, n + 1)]))
    z, y, x = find_objects(cc == biggest)[0]
    return (z.start + z.stop) // 2, (y.start + y.stop) // 2, (x.start + x.stop) // 2


def normalise(vol, rng):
    fg = vol > 0
    if not fg.any():
        return np.zeros_like(vol)
    ref = vol[fg]
    if ref.size > 1_000_000:
        ref = ref[rng.choice(ref.size, 1_000_000, replace=False)]
    lo, hi = np.percentile(ref, [0.5, 99.5])
    v = np.clip(vol, lo, hi)
    m, s = v[fg].mean(), v[fg].std()
    return np.where(fg, (v - m) / s, 0.0).astype(np.float32) if s > 1e-6 else np.zeros_like(vol)


def nonzero_z(x, eps=1e-5):
    """z-score over the nonzero voxels, zeros stay zero (BrainIAC's NormalizeIntensity(nonzero=True))."""
    m = (x != 0).float()
    n = m.sum().clamp(min=1.0)
    mu = (x * m).sum() / n
    sd = (((x - mu) * m) ** 2).sum().div(n).sqrt().clamp(min=eps)
    return (x - mu) * m / sd


class Encoder:
    """Frozen BrainIAC (Tak et al., Nat. Neurosci. 2026; github.com/AIM-KannLab/BrainIAC): MONAI ViT-B, 96^3 input, patch 16."""

    def __init__(self, device, ckpt):
        from monai.networks.nets import ViT
        vit = ViT(in_channels=1, img_size=RESIZE, patch_size=(16, 16, 16), hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = {k[len("backbone."):]: v for k, v in sd.get("state_dict", sd).items() if k.startswith("backbone.")}
        missing, unexpected = vit.load_state_dict(sd, strict=False)
        # newer MONAI ViT blocks also hold cross-attention weights, which the checkpoint lacks; they are unused without a context
        assert sd and not unexpected and all("cross_attn" in k for k in missing), (missing[:5], unexpected[:5])
        self.net = vit.eval().requires_grad_(False).to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, vol):
        x = F.interpolate(torch.from_numpy(vol)[None, None].to(self.device), size=RESIZE, mode="trilinear", align_corners=False)
        return self.net(nonzero_z(x))[0][0, 0].cpu()          # CLS token


def encode_patient(scans, enc, rng):
    """scans: [{"time": weeks, "paths": {sequence: nifti}}] in time order; every crop is centered on the brain."""
    feats = []
    for s in scans:
        centre = brain_centre(s["paths"]["CT1"])
        f = []
        for seq in ("CT1", "T1", "T2", "FLAIR"):
            vol, sp = load_ras(s["paths"][seq])
            f.append(enc(normalise(crop(resample(vol, sp, 1), centre), rng)))
        feats.append(torch.stack(f))
    return {"feats": torch.stack(feats), "time": torch.tensor([s["time"] for s in scans], dtype=torch.float32)}


def lumiere_scans(root, pid):
    scans = []
    for wf in sorted(w for w in os.listdir(os.path.join(root, pid)) if "week" in w):
        d = os.path.join(root, pid, wf, "HD-GLIO-AUTO-segmentation", "registered")
        paths = {k: os.path.join(d, v) for k, v in LUMIERE_SEQ.items()}
        if not (os.path.exists(os.path.join(d, "segmentation.nii.gz")) and all(os.path.exists(p) for p in paths.values())):
            continue
        a = re.match(r"week-(\d+)(?:-(\d+))?$", wf)
        week = float(a.group(1)) + (float(a.group(2)) / 100.0 if a.group(2) else 0.0) if a else 0.0
        scans.append({"time": week, "paths": paths})
    return sorted(scans, key=lambda s: s["time"])


def mu_scans(root, pid, days):
    scans = []
    for tp in glob.glob(os.path.join(root, pid, "Timepoint_*")):
        n = int(re.search(r"_(\d+)$", tp).group(1))
        base = f"{pid}_Timepoint_{n}"
        paths = {k: os.path.join(tp, f"{base}_brain_{v}.nii.gz") for k, v in MU_SEQ.items()}
        if not (os.path.exists(os.path.join(tp, f"{base}_tumorMask.nii.gz")) and all(os.path.exists(p) for p in paths.values())):
            continue
        scans.append({"n": n, "day": days.get((pid, n)), "paths": paths})
    scans.sort(key=lambda s: s["n"])
    known = [s["day"] for s in scans if s["day"] is not None]
    for s in scans:
        s["time"] = (s["day"] - min(known)) / 7.0 if known and s["day"] is not None else (s["n"] - scans[0]["n"]) * MU_FALLBACK_WEEKS
    t0 = min(s["time"] for s in scans) if scans else 0.0
    for s in scans:
        s["time"] -= t0
    return sorted(scans, key=lambda s: s["time"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cohort", choices=["lumiere", "mu"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--brainiac", required=True, help="BrainIAC.ckpt from the authors (github.com/AIM-KannLab/BrainIAC)")
    ap.add_argument("--days", help="MU: CSV with Patient_ID, Timepoint, DaysFromDx")
    ap.add_argument("--patients", nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = Encoder(dev, a.brainiac)
    days = {}
    if a.days:
        days = {(r["Patient_ID"], int(r["Timepoint"])): float(r["DaysFromDx"]) for r in csv.DictReader(open(a.days))}
    prefix = "Patient-" if a.cohort == "lumiere" else "PatientID_"
    pids = a.patients or sorted(p for p in os.listdir(a.root) if p.startswith(prefix))
    for i, pid in enumerate(pids, 1):
        dst = os.path.join(a.out, f"{pid}.pt")
        if os.path.exists(dst):
            continue
        scans = lumiere_scans(a.root, pid) if a.cohort == "lumiere" else mu_scans(a.root, pid, days)
        if not scans:
            print(f"skip {pid}: no complete scans")
            continue
        torch.save(encode_patient(scans, enc, np.random.RandomState(zlib.crc32(pid.encode()))), dst)
        print(f"[{i}/{len(pids)}] {pid}: {len(scans)} scans", flush=True)


if __name__ == "__main__":
    main()

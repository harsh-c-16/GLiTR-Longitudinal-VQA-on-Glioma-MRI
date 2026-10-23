"""Trend and net-change labels from per-component segmentation volumes (same rule for both cohorts).

Each consecutive scan pair is a rise, a fall or no change. Trend (Q1-3): Fluctuating if the series has a rise and a fall, else
Increase / Decrease, else Stable. Net change (Q4-6): first vs. last scan. Thresholds (rise / fall): enhancing tumor 40% / 65%
(RANO 2.0), edema 15% / 15% (Rudie 2022), necrosis 25% (Nowosielski 2019) / 65% (RANO fallback).
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

COMPONENTS = ("necrosis", "tumor", "edema")                 # trend question ids 1-3, net question ids 4-6
LABELS = {-1: "Decrease", 0: "Stable", 1: "Increase", 2: "Fluctuating"}
RULE = {"necrosis": (0.25, 0.65), "tumor": (0.40, 0.65), "edema": (0.15, 0.15)}
THRESH = {"lumiere": RULE, "mu": RULE}
MU_LABEL_ID = {"necrosis": 1, "edema": 2, "tumor": 3}       # BraTS-2024 NETC / SNFH / ET (4 = cavity, ignored)
# LUMIERE_data.csv swaps the names of DeepBraTumIA ids 1 and 2: id 1 ("Necrosis") is the enhancing core, id 2 ("Contrast-enhancing")
# the necrotic core (CT1 uptake 1.51x vs 0.79x normal brain, 266/266 visits). Names below follow the image content.
LUMIERE_NAME = {"Necrosis": "tumor", "Contrast-enhancing": "necrosis", "Edema": "edema"}


def step(prev, cur, comp, cohort):
    """+1 rise, -1 fall, 0 no change between two volumes of one component."""
    up, down = THRESH[cohort][comp]
    if prev <= 0:
        return 1 if cur > 0 else 0
    r = (cur - prev) / prev
    return 1 if r > up else -1 if r < -down else 0


def trend(vols, comp, cohort):
    s = [step(a, b, comp, cohort) for a, b in zip(vols[:-1], vols[1:])]
    return 2 if 1 in s and -1 in s else 1 if 1 in s else -1 if -1 in s else 0


def net(vols, comp, cohort):
    if len(vols) < 2:
        return 0
    return step(vols[0], vols[-1], comp, cohort)


def rows_for(pid, series, cohort):
    out = []
    for i, comp in enumerate(COMPONENTS):
        v = series[comp]
        out.append((pid, i + 1, comp, trend(v, comp, cohort) if len(v) >= 2 else 0))
        out.append((pid, i + 4, comp, net(v, comp, cohort)))
    return out


def tp_key(name):
    m = re.match(r"week-(\d+)(?:-(\d+))?", str(name))
    return (int(m.group(1)), int(m.group(2) or 0)) if m else (10 ** 6, 0)


def lumiere_labels(volumes_csv):
    """volumes_csv: LUMIERE's DeepBraTumIA voxel counts (Patient, Timepoint, Label, NumberOfVoxels)."""
    df = pd.read_csv(volumes_csv)
    df.columns = df.columns.str.strip()
    df["NumberOfVoxels"] = pd.to_numeric(df["NumberOfVoxels"], errors="coerce").fillna(0)
    rows = []
    for pid, g in df.groupby(df["Patient"].str.strip()):
        tps = sorted(g["Timepoint"].unique(), key=tp_key)
        series = {}
        for name, comp in LUMIERE_NAME.items():
            by_tp = g[g["Label"].str.strip() == name].set_index("Timepoint")["NumberOfVoxels"]
            series[comp] = [float(by_tp.get(t, 0.0)) for t in tps]
        rows += rows_for(pid, series, "lumiere")
    return rows


def mu_labels(root, min_scans=2):
    import nibabel as nib
    rows = []
    for pdir in sorted(glob.glob(os.path.join(root, "PatientID_*"))):
        tps = sorted(glob.glob(os.path.join(pdir, "Timepoint_*")), key=lambda p: int(re.search(r"_(\d+)$", p).group(1)))
        series = {c: [] for c in COMPONENTS}
        for tp in tps:
            mask = glob.glob(os.path.join(tp, "*_tumorMask.nii.gz"))
            if not mask:
                continue
            counts = np.bincount(nib.load(mask[0]).get_fdata().astype(np.int32).ravel(), minlength=5)
            for c in COMPONENTS:
                series[c].append(int(counts[MU_LABEL_ID[c]]))
        if len(series["necrosis"]) >= min_scans:
            rows += rows_for(os.path.basename(pdir), series, "mu")
    return sorted(rows, key=lambda r: (r[0], r[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cohort", choices=["lumiere", "mu"])
    ap.add_argument("--volumes", help="LUMIERE_data.csv (lumiere)")
    ap.add_argument("--root", help="MU-Glioma-Post root (mu)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = lumiere_labels(a.volumes) if a.cohort == "lumiere" else mu_labels(a.root)
    pd.DataFrame(rows, columns=["Patient_ID", "Question_ID", "Component", "Trend_Direction"]).to_csv(a.out, index=False)
    d = pd.Series([r[3] for r in rows]).map(LABELS).value_counts().to_dict()
    print(f"wrote {a.out}: {len(rows)} rows, {len({r[0] for r in rows})} patients, {d}")


if __name__ == "__main__":
    main()

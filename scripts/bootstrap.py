"""Paired patient-bootstrap intervals for HM-F1, plus macro AUROC; per-class precision / recall and confusion matrix for the
reference run and any run ahead of it.

Every run in runs/<cohort>/<name>/s<seed>/oof.csv is compared with the reference (--ref). Both runs of a comparison are scored on
the same patient resamples, averaged over seeds, so the interval measures patient-sampling noise, not seed noise (the seed SD is
printed next to it).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glitr.evaluate import CLASSES, auroc_csv

SEEDS = (42, 43, 44, 45, 46)
LABEL = {-1: "Decrease", 0: "Stable", 1: "Increase", 2: "Fluctuating"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def hm_from_counts(tp, fp, fn):
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    return np.where((f1 > 0).all(-1), 4.0 / np.sum(1.0 / np.maximum(f1, 1e-12), -1), 0.0)


def samples(cohort, questions):
    """The scored samples: every (patient, component, question) of a patient that is in the folds file."""
    gt = pd.read_csv(f"{ROOT}/configs/gt_{cohort}.csv")
    gt = gt[gt.Patient_ID.isin(set(pd.read_csv(f"{ROOT}/configs/folds_{cohort}.csv").Patient_ID))]
    if questions:
        gt = gt[gt.Question_ID.isin(questions)]
    return {(r.Patient_ID, r.Component.lower(), int(r.Question_ID)): LABEL[int(r.Trend_Direction)] for r in gt.itertuples()}


def encode(path, truth, keys):
    df = pd.read_csv(path)
    pred = {(r.Patient_ID, r.Component.lower(), int(r.Question_ID)): r.Pred_Label for r in df.itertuples()}
    y = np.array([CLASSES.index(truth[k]) for k in keys])
    p = np.array([CLASSES.index(pred[k]) for k in keys])
    return y, p


def runs(results, cohort):
    """{run name: seed files} of every run that has all five seeds."""
    d, out = f"{results}/{cohort}", {}
    for m in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        files = [f"{d}/{m}/s{s}/oof.csv" for s in SEEDS]
        if all(os.path.exists(f) for f in files):
            out[m] = files
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--out", default="bootstrap_report")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--ref", default="full")
    ap.add_argument("--pairs", nargs="*", default=[], help="extra comparisons A:B (run names)")
    ap.add_argument("--questions", type=int, nargs="*")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    text, J = [], {}
    for cohort in ("lumiere", "mu"):
        truth = samples(cohort, a.questions)
        keys = sorted(truth)
        patients, inv = np.unique(np.array([k[0] for k in keys]), return_inverse=True)
        draws = rng.integers(0, len(patients), size=(a.boot, len(patients)))
        W = np.stack([np.bincount(d, minlength=len(patients)) for d in draws])[:, inv].astype(np.float32)
        paths = runs(a.results, cohort)
        stats = {}
        for name, files in paths.items():
            hms, macs, pcs, prs, rcs, cms, boot = [], [], [], [], [], [], np.zeros(a.boot)
            for f in files:
                y, p = encode(f, truth, keys)
                Y = np.eye(4)[y]
                P = np.eye(4)[p]
                TP = Y * P
                boot += hm_from_counts(W @ TP, W @ (P - TP), W @ (Y - TP)) / len(files)
                tp, fp, fn = TP.sum(0), (P - TP).sum(0), (Y - TP).sum(0)
                hms.append(float(hm_from_counts(tp, fp, fn)))
                pcs.append(2 * tp / np.maximum(2 * tp + fp + fn, 1))
                macs.append(float(np.mean(pcs[-1])))
                prs.append(tp / np.maximum(tp + fp, 1))
                rcs.append(tp / np.maximum(tp + fn, 1))
                cms.append(Y.T @ P)
            stats[name] = dict(hms=hms, hm=np.mean(hms), sd=np.std(hms, ddof=1), boot=boot, pc=np.mean(pcs, 0), macro=np.mean(macs), macro_sd=np.std(macs, ddof=1),
                               prec=np.mean(prs, 0), prec_sd=np.std(prs, 0, ddof=1), rec=np.mean(rcs, 0), rec_sd=np.std(rcs, 0, ddof=1), cm=np.mean(cms, 0))
            au = [None] if a.questions else [auroc_csv(f) for f in files]              # scores are not re-pooled for subsets
            if all(au):
                m = np.array([x[0] for x in au])
                stats[name].update(auroc=float(m.mean()), auroc_sd=float(m.std(ddof=1)))
        text.append(f"\n===== {cohort.upper()}  n={len(keys)} samples, {len(patients)} patients, bootstrap B={a.boot}"
                    + (f", questions {a.questions}" if a.questions else "") + " =====")
        text.append(f"{'run':24s} HM-F1 +-seed-SD   95% CI           macro-F1   AUROC   per-class F1 D/S/I/F")
        J[cohort] = dict(n=len(keys), patients=len(patients), methods={}, pairs={})
        for k in sorted(stats, key=lambda k: -stats[k]["hm"]):
            s = stats[k]
            lo, hi = np.percentile(s["boot"], [2.5, 97.5])
            au = f"{s['auroc']:.3f}" if "auroc" in s else "  -  "
            text.append(f"{k:24s} {s['hm']:.3f} +-{s['sd']:.3f}   [{lo:.3f}, {hi:.3f}]   {s['macro']:.3f}     {au}   " + " ".join(f"{v:.3f}" for v in s["pc"]))
            J[cohort]["methods"][k] = dict(hm=s["hm"], hm_sd=s["sd"], lo=float(lo), hi=float(hi), macro=s["macro"], macro_sd=s["macro_sd"],
                                           pc=[float(x) for x in s["pc"]], **{x: s[x] for x in ("auroc", "auroc_sd") if x in s},
                                           **{x: np.round(s[x], 4).tolist() for x in ("hms", "prec", "prec_sd", "rec", "rec_sd", "cm")})
        pairs = [(x, a.ref) for x in sorted(stats) if x != a.ref] + [tuple(p.split(":")) for p in a.pairs]
        text.append("-- paired differences in HM-F1 (A minus B), 95% patient-bootstrap interval, P(A>B); * = interval excludes 0")
        for x, y in dict.fromkeys(pairs):
            if x in stats and y in stats:
                d = stats[x]["boot"] - stats[y]["boot"]
                lo, hi = np.percentile(d, [2.5, 97.5])
                text.append(f"{x:22s} - {y:22s} {stats[x]['hm'] - stats[y]['hm']:+.3f}  [{lo:+.3f}, {hi:+.3f}]  P(A>B)={np.mean(d > 0):.2f}{'  *' if lo > 0 or hi < 0 else ''}")
                J[cohort]["pairs"][f"{x}|{y}"] = dict(diff=float(stats[x]["hm"] - stats[y]["hm"]), lo=float(lo), hi=float(hi), p=float(np.mean(d > 0)))
        R = stats.get(a.ref)
        ahead = [k for k in sorted(stats, key=lambda k: -stats[k]["hm"]) if R and k != a.ref and (stats[k]["hm"] > R["hm"] or stats[k]["pc"][3] > R["pc"][3])]
        text.append(f"-- {a.ref} and every run ahead of it on HM-F1 or Fluctuating F1: per-class precision / recall D/S/I/F (mean +- seed SD) and the "
                    "confusion matrix (mean over seeds; rows truth D/S/I/F, columns prediction D/S/I/F)")
        for k in ([a.ref] if R else []) + ahead:
            s = stats[k]
            text.append(f"{k}  HM-F1 {s['hm']:.3f} +-{s['sd']:.3f}, Fluctuating F1 {s['pc'][3]:.3f}")
            text += [f"   {n:9s} " + "  ".join(f"{m:.3f}+-{v:.3f}" for m, v in zip(s[x], s[x + '_sd'])) for n, x in (("precision", "prec"), ("recall", "rec"))]
            text += [f"   {c}  " + " ".join(f"{v:6.1f}" for v in row) for c, row in zip("DSIF", s["cm"])]
    open(a.out + ".txt", "w").write("\n".join(text) + "\n")
    json.dump(J, open(a.out + ".json", "w"), indent=1)
    print("\n".join(text))


if __name__ == "__main__":
    main()

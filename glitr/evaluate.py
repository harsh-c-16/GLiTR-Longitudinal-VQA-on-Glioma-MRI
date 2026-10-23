"""Metrics on pooled out-of-fold predictions: macro-F1 = mean of the four per-class F1 scores, HM-F1 = their harmonic mean
(0 if any class has F1 0), and macro AUROC.
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

CLASSES = ["Decrease", "Stable", "Increase", "Fluctuating"]


def per_class_f1(y_true, y_pred, labels=CLASSES):
    return f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)


def macro_hm(f1):
    return float(np.mean(f1)), float(len(f1) / sum(1.0 / max(float(v), 1e-9) for v in f1))


def score_csv(path, exclude=()):
    df = pd.read_csv(path)
    df = df[~df["Patient_ID"].isin(exclude)]
    f = per_class_f1(df["GT_Label"], df["Pred_Label"])
    return (*macro_hm(f), f, len(df))


def auroc_csv(path, exclude=()):
    """One-vs-rest AUROC per class from the P_<class> columns (None if absent) and its macro mean; folds are pooled."""
    df = pd.read_csv(path)
    cols = [f"P_{c}" for c in CLASSES]
    if not all(c in df for c in cols):
        return None
    df = df[~df["Patient_ID"].isin(exclude)]
    a = [roc_auc_score(df["GT_Label"] == c, df[f"P_{c}"]) if 0 < (df["GT_Label"] == c).sum() < len(df) else np.nan for c in CLASSES]
    return float(np.nanmean(a)), np.array(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--exclude", nargs="*", default=[], help="patients to leave out of the scoring")
    a = ap.parse_args()
    res = [score_csv(os.path.join(a.run_dir, f"s{s}", "oof.csv"), a.exclude) for s in a.seeds]
    macro, hm = np.array([r[0] for r in res]), np.array([r[1] for r in res])
    pc = np.mean([r[2] for r in res], axis=0)
    sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    print(f"{a.run_dir}: n={res[0][3]} samples, seeds {a.seeds}")
    print(f"  macro-F1 {macro.mean():.4f} +- {sd(macro):.4f} | HM-F1 {hm.mean():.4f} +- {sd(hm):.4f}")
    print("  per-class F1 " + "  ".join(f"{c} {v:.3f}" for c, v in zip(CLASSES, pc)))
    au = [auroc_csv(os.path.join(a.run_dir, f"s{s}", "oof.csv"), a.exclude) for s in a.seeds]
    if all(au):
        m = np.array([x[0] for x in au])
        print(f"  macro AUROC (one-vs-rest) {m.mean():.4f} +- {sd(m):.4f} | per class " +
              "  ".join(f"{c} {v:.3f}" for c, v in zip(CLASSES, np.nanmean([x[1] for x in au], axis=0))))


if __name__ == "__main__":
    main()

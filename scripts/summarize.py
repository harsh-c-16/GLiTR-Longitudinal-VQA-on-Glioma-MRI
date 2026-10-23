"""HM-F1, macro-F1 and macro AUROC (mean +- SD over seeds 42-46) of every finished run in runs/<cohort>/<name>/s<seed>/oof.csv."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glitr.evaluate import auroc_csv, score_csv

root = sys.argv[1] if len(sys.argv) > 1 else "runs"
sd = lambda x: float(np.std(x, ddof=1))
print(f"{'cohort':8s} {'arm':16s} {'HM-F1':>13s} {'macro-F1':>13s} {'AUROC':>13s}")
for cohort in ("lumiere", "mu"):
    for arm in sorted(os.listdir(os.path.join(root, cohort))) if os.path.isdir(os.path.join(root, cohort)) else []:
        files = [os.path.join(root, cohort, arm, f"s{s}", "oof.csv") for s in (42, 43, 44, 45, 46)]
        if not all(os.path.exists(f) for f in files):
            continue
        sc = [score_csv(f) for f in files]
        au = [auroc_csv(f) for f in files]
        hm, ma = [x[1] for x in sc], [x[0] for x in sc]
        a = f"{np.mean([x[0] for x in au]):.3f} +- {sd([x[0] for x in au]):.3f}" if all(au) else "-"
        print(f"{cohort:8s} {arm:16s} {np.mean(hm):.3f} +- {sd(hm):.3f} {np.mean(ma):.3f} +- {sd(ma):.3f} {a:>13s}")

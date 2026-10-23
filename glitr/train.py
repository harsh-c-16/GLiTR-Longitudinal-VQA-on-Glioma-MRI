"""Patient-grouped 5-fold cross-validation with nested checkpoint selection.

Each outer fold's training patients are split into inner-train / inner-validation. The checkpoint (best inner HM-F1) and early
stopping use inner-validation only; the outer fold is scored once. Hyperparameters come from --hp, --hp-file or --tune-trials
(Optuna inside the outer training patients); none are defaulted. Writes <out>/s<seed>/oof.csv.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import (Trajectories, BalancedBatches, build_folds, collate, folds_from_file, cls_of, inner_folds, inner_split,
                   load_features, load_gt, tokenize)
from .evaluate import CLASSES, auroc_csv, macro_hm, per_class_f1, score_csv
from .model import FocalLoss, Model, bert_path, class_probs, decode

EVAL_EVERY = 5


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def to(batch, dev):
    return {k: v.to(dev) for k, v in batch.items()}


def forward(model, b):
    return model(b["feats"], b["time"], b["tmask"], b["comp"], b["q_ids"], b["q_mask"])


@torch.no_grad()
def predict(model, ds, dev, return_probs=False):
    model.eval()
    pred, truth, probs = [], [], []
    for b in DataLoader(ds, batch_size=32, collate_fn=collate):
        b = to(b, dev)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            bl, tl = forward(model, b)
        pred += decode(bl, tl).cpu().tolist()
        truth += b["cls"].cpu().tolist()
        probs += class_probs(bl, tl).cpu().tolist()
    return (np.array(pred), np.array(truth), np.array(probs)) if return_probs else (np.array(pred), np.array(truth))


def score(truth, pred):
    return macro_hm(per_class_f1(truth, pred, [0, 1, 2, 3]))


def fit(args, hp, train_ds, val_ds, dev, seed, report=None):
    set_seed(seed)
    model = Model(hp["dropout"], feat=args.feat).to(dev)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    focal = FocalLoss(hp["focal_gamma"])
    loader = DataLoader(train_ds, batch_sampler=BalancedBatches(train_ds.by_class, seed), collate_fn=collate)
    best, best_state, stale = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for b in loader:
            b = to(b, dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                bl, tl = forward(model, b)
                change = b["binary"] == 1
                loss = focal(bl, b["binary"]) + (focal(tl[change], b["tri"][change]) if change.any() else 0.0)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        if epoch % EVAL_EVERY:
            continue
        pred, truth = predict(model, val_ds, dev)
        hm = score(truth, pred)[1]
        sched.step(hm)
        if report is not None and report(hm, epoch):
            break
        if hm > best:
            best, stale = hm, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += EVAL_EVERY
            if stale >= args.patience:
                break
    return best_state, best


def tune(args, train_p, feats, gt, tok, dev, seed):
    """Optuna (TPE, median pruner) on --tune-inner-folds patient-grouped splits of one outer training set; score = mean best inner HM-F1."""
    import optuna
    splits = [(mk(args, feats, gt, a, tok), mk(args, feats, gt, b, tok)) for a, b in inner_folds(train_p, args.tune_inner_folds, seed)]

    def objective(trial):
        hp = {"lr": trial.suggest_float("lr", 1e-5, 5e-4, log=True),
              "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
              "dropout": trial.suggest_float("dropout", 0.1, 0.5),
              "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 5.0)}
        scores = []
        for i, (tr, va) in enumerate(splits):
            pruned = []

            def report(hm, epoch):
                trial.report(hm, epoch)
                pruned.append(trial.should_prune())
                return pruned[-1]
            _, best = fit(args, hp, tr, va, dev, seed, report if i == 0 else None)
            if pruned and pruned[-1]:
                raise optuna.TrialPruned()
            scores.append(best)
        return float(np.mean(scores))
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=8))
    study.optimize(objective, n_trials=args.tune_trials)
    return study.best_params


def fold_hp(args, base, k, train_p, feats, gt, tok, dev, seed):
    tuned = json.load(open(args.hp_file)).get(args.cohort, {}).get(f"fold{k}") if args.hp_file else None
    if tuned:
        return {**base, **tuned}
    if args.tune_trials:
        best = tune(args, train_p, feats, gt, tok, dev, seed)
        json.dump(best, open(os.path.join(args.out, f"s{args.seed}", f"hp_fold{k}.json"), "w"), indent=1)
        return {**base, **best}
    if base:
        return base
    raise SystemExit(f"no hyperparameters for fold {k}: pass --hp, --hp-file with --cohort, or --tune-trials")


def mk(args, feats, gt, patients, tok):
    return Trajectories(feats, gt, patients, tok, args.ablation)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cv-seed", type=int, default=42)
    ap.add_argument("--folds-file", help="CSV (Patient_ID, fold) with explicit outer folds, e.g. configs/folds_lumiere.csv")
    ap.add_argument("--inner-frac", type=float, default=0.2)
    ap.add_argument("--hp", help="json with lr, weight_decay, dropout, focal_gamma")
    ap.add_argument("--tune-trials", type=int, default=0, help="Optuna trials per outer fold, on inner cross-validation")
    ap.add_argument("--tune-inner-folds", type=int, default=3, help="inner cross-validation folds per tuning trial")
    ap.add_argument("--only-fold", type=int, help="run a single outer fold (parallel tuning); pooled scores are skipped")
    ap.add_argument("--hp-file", help="tuned hyperparameters per cohort and outer fold, e.g. configs/hp.json")
    ap.add_argument("--cohort", choices=["lumiere", "mu"], help="key in --hp-file")
    ap.add_argument("--ablation", choices=["none", "no_images", "no_order"], default="none",
                    help="input ablation: zero every image feature, or shuffle the scans of a series and zero the times")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=25)
    args = ap.parse_args()
    if args.hp_file and not args.cohort:
        ap.error("--hp-file needs --cohort")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = os.path.join(args.out, f"s{args.seed}")
    os.makedirs(out, exist_ok=True)
    gt = load_gt(args.gt)
    patients = sorted(p[:-3] for p in os.listdir(args.features) if p.endswith(".pt") and p[:-3] in gt)
    feats = load_features(args.features, patients)
    args.feat = feats[patients[0]]["feats"].shape[-1]
    from transformers import AutoTokenizer
    tok = tokenize(AutoTokenizer.from_pretrained(bert_path()))
    base_hp = json.load(open(args.hp)) if args.hp else {}

    frames = []
    folds = folds_from_file(args.folds_file, patients) if args.folds_file else build_folds(gt, patients, args.folds, args.cv_seed)
    for k, (train_p, test_p) in enumerate(folds, 1):
        if args.only_fold and k != args.only_fold:
            continue
        csv = os.path.join(out, f"fold{k}.csv")
        if not os.path.exists(csv):
            seed = args.seed + k
            hp = fold_hp(args, base_hp, k, train_p, feats, gt, tok, dev, seed)
            test = mk(args, feats, gt, test_p, tok)
            tr_p, va_p = inner_split(train_p, args.inner_frac, seed)
            state, best = fit(args, hp, mk(args, feats, gt, tr_p, tok), mk(args, feats, gt, va_p, tok), dev, seed)
            model = Model(hp["dropout"], feat=args.feat).to(dev)
            model.load_state_dict(state)
            pred, truth, prob = predict(model, test, dev, return_probs=True)
            rows = [(p, c, q, CLASSES[cls_of(d)], CLASSES[y], *np.round(pr, 6)) for (p, c, q, d), y, pr in zip(test.s, pred, prob)]
            pd.DataFrame(rows, columns=["Patient_ID", "Component", "Question_ID", "GT_Label", "Pred_Label",
                                        *[f"P_{c}" for c in CLASSES]]).to_csv(csv, index=False)
            print(f"fold {k}: selection HM-F1 {best:.3f} | outer macro-F1 {score(truth, pred)[0]:.3f} | hp {hp}", flush=True)
        frames.append(pd.read_csv(csv))
    if args.only_fold:
        return
    oof = pd.concat(frames).sort_values(["Patient_ID", "Component", "Question_ID"])
    oof.to_csv(os.path.join(out, "oof.csv"), index=False)
    macro, hm, f, n = score_csv(os.path.join(out, "oof.csv"))
    print(f"pooled out-of-fold ({n} samples): macro-F1 {macro:.4f} HM-F1 {hm:.4f} | " + " ".join(f"{c} {v:.3f}" for c, v in zip(CLASSES, f)))
    au = auroc_csv(os.path.join(out, "oof.csv"))
    if au:
        print(f"  macro AUROC (one-vs-rest) {au[0]:.4f} | " + " ".join(f"{c} {v:.3f}" for c, v in zip(CLASSES, au[1])))


if __name__ == "__main__":
    main()

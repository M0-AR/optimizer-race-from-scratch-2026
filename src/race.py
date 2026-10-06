"""Part 8: the race. 17 LRs x 5 seeds x 400 steps, median (middle) of 5 reported.

LR grid: logspace(1e-4, 1.0, 17). Threshold for 'works': final loss < 0.1.
Also records steps-to-0.1, test accuracy, gradient spread (top1%/bottom1%).
"""
import numpy as np
import json
from .mlp import init_params, train, loss_and_grads, accuracy
from .optimizers import make_optimizer

LAYERS = [64, 32, 32, 32, 32, 32, 10]
LR_GRID = list(np.logspace(-4, 0, 17))
SEEDS = [0, 1, 2, 3, 4]


def gradient_spread(params, X, y, sample=512):
    """Ratio |g|_top1% / |g|_bottom1% on a sample batch (video: ~22000x)."""
    idx = np.random.default_rng(0).integers(0, len(X), min(sample, len(X)))
    _, grads, _ = loss_and_grads(params, X[idx], y[idx])
    flat = np.abs(np.concatenate([g.ravel() for g in grads]))
    flat = flat[flat > 0]
    if len(flat) < 100:
        return float("nan")
    lo = np.percentile(flat, 1)
    hi = np.percentile(flat, 99)
    return float(hi / max(lo, 1e-18))


def race_one_optimizer(name, Xtr, ytr, Xte, yte, steps=400, batch=32, opt_kw=None):
    opt_kw = opt_kw or {}
    rows = []
    for lr in LR_GRID:
        finals, accs = [], []
        for s in SEEDS:
            p0 = init_params(LAYERS, seed=100 + s)
            opt = make_optimizer(name, **opt_kw)
            p, hist = train(p0, opt, Xtr, ytr, lr, steps, batch, seed=s)
            fl, _, _ = loss_and_grads(p, Xtr[:512], ytr[:512])
            finals.append(float(loss_and_grads(p, Xtr, ytr)[0]) if len(Xtr) < 2000 else float(np.mean(hist[-10:])))
            accs.append(float(accuracy(p, Xte, yte)))
        finals_sorted = sorted(finals)
        accs_sorted = sorted(accs)
        rows.append({"lr": float(lr), "median_loss": finals_sorted[2],
                     "median_acc": accs_sorted[2], "all_loss": finals, "all_acc": accs})
    best = min(rows, key=lambda r: r["median_loss"])
    n_work = sum(1 for r in rows if r["median_loss"] < 0.1)
    return {"optimizer": name, "best": best, "n_working": n_work, "grid": rows}


def run_full(Xtr, ytr, Xte, yte, steps=400, optimizers=("sgd", "momentum", "rmsprop", "adam", "muon")):
    out = {}
    for name in optimizers:
        print(f"[race] {name} ...", flush=True)
        out[name] = race_one_optimizer(name, Xtr, ytr, Xte, yte, steps)
        print(f"  best loss={out[name]['best']['median_loss']:.4f} "
              f"lr={out[name]['best']['lr']:.4g} acc={out[name]['best']['median_acc']:.3f} "
              f"working={out[name]['n_working']}/17", flush=True)
    return out

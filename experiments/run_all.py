"""End-to-end reproduction: valley -> digits race -> live market -> figures + results.json.

Usage:
  python3 experiments/run_all.py --quick   # ~2 min on laptop (reduced grid)
  python3 experiments/run_all.py --full    # full 17x5x400 protocol (~10-20 min numpy)
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.valley import run as valley_run, loss as valley_loss
from src import optimizers as O
from src.data import load_digits_split
from src.mlp import init_params, train, loss_and_grads, accuracy
from src.race import race_one_optimizer, run_full, gradient_spread, LAYERS, LR_GRID, SEEDS

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def valley_table():
    print("== Valley (Part 1/3/4/5) ==")
    rows = {}
    # video anchor points
    cfgs = {"sgd": ("sgd", 0.019), "momentum": ("momentum", 0.019),
            "rmsprop": ("rmsprop", 0.01), "adam": ("adam", 0.3)}
    for k, (name, lr) in cfgs.items():
        opt = O.make_optimizer(name)
        p, losses = valley_run(opt, lr, steps=1000)
        rows[k] = {"lr": lr, "final_loss": float(losses[-1]), "steps": len(losses)}
        print(f"  {k} lr={lr}: loss={losses[-1]:.5f} in {len(losses)} steps")
    # blow-up check: sgd lr=0.021 must diverge
    _, losses = valley_run(O.make_optimizer("sgd"), 0.021, steps=500)
    rows["sgd_blowup_0.021"] = {"final_loss": float(losses[-1]) if np.isfinite(losses[-1]) else "inf",
                                "diverged": bool(not np.isfinite(losses[-1]) or losses[-1] > 1e6)}
    print(f"  sgd lr=0.021 diverged={rows['sgd_blowup_0.021']['diverged']}")
    # LR robustness count over 101 LRs 1e-4..10
    lrs = list(np.logspace(-4, 1, 101))
    for name in ["sgd", "momentum", "rmsprop", "adam"]:
        n = 0
        for lr in lrs:
            _, losses = valley_run(O.make_optimizer(name), lr, steps=1000)
            if np.isfinite(losses[-1]) and losses[-1] < 1e-3:
                n += 1
        rows[f"{name}_robust_101"] = n
        print(f"  {name} robust: {n}/101")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    full = a.full and not a.quick
    t0 = time.time()
    out = {"protocol": "full-17x5x400" if full else "quick",
           "valley": valley_table()}
    (Xtr, ytr), (Xte, yte) = load_digits_split()
    print(f"digits: train {Xtr.shape} test {Xte.shape}")
    p0 = init_params(LAYERS, seed=100)
    out["gradient_spread_top1/bottom1"] = gradient_spread(p0, Xtr, ytr)
    print(f"gradient spread: {out['gradient_spread_top1/bottom1']:.0f}x (video ~22000x)")
    if full:
        out["race"] = run_full(Xtr, ytr, Xte, yte, steps=400)
    else:
        # quick: 5 LRs x 3 seeds x 200 steps
        from src.race import SEEDS as _S
        import src.race as R
        R_LR = [1e-4, 1e-3, 1e-2, 0.05, 0.2]
        race = {}
        for name in ["sgd", "momentum", "rmsprop", "adam", "muon"]:
            rows = []
            for lr in R_LR:
                finals, accs = [], []
                for s in [0, 1, 2]:
                    pp0 = init_params(LAYERS, seed=100 + s)
                    p, _ = train(pp0, O.make_optimizer(name), Xtr, ytr, lr, 200, 32, seed=s)
                    finals.append(float(loss_and_grads(p, Xtr, ytr)[0]))
                    accs.append(float(accuracy(p, Xte, yte)))
                rows.append({"lr": lr, "median_loss": sorted(finals)[1], "median_acc": sorted(accs)[1]})
            best = min(rows, key=lambda r: r["median_loss"])
            race[name] = {"best": best, "grid": rows,
                          "n_working": sum(1 for r in rows if r["median_loss"] < 0.3)}
            print(f"[quick-race] {name}: best loss={best['median_loss']:.4f} acc={best['median_acc']:.3f}")
        out["race"] = race
    # live market verification (always, ~1-2 min)
    try:
        from src.market_live import run_market_experiment
        out["market_live"] = run_market_experiment(steps=400 if full else 200)
    except Exception as e:
        out["market_live"] = {"error": str(e)}
        print(f"[market] failed: {e}")
    out["elapsed_sec"] = time.time() - t0
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote results/results.json in {out['elapsed_sec']:.0f}s")
    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(out["race"].keys())
        losses = [out["race"][k]["best"]["median_loss"] for k in names]
        plt.figure()
        plt.bar(names, losses)
        plt.ylabel("best median loss (digits)")
        plt.title("Optimizer race — best of grid (median of seeds)")
        plt.savefig(os.path.join(FIG, "race_best_loss.png"), dpi=120)
        print("wrote figures/race_best_loss.png")
    except Exception as e:
        print(f"figures failed: {e}")


if __name__ == "__main__":
    main()

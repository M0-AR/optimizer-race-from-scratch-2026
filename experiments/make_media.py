"""Generate all visual media from measured code (no hand-drawn numbers).

Outputs (all regenerated from scratch on every run):
  figures/valley_race.gif      - animated SGD-vs-Momentum-vs-Adam descent on the valley
  figures/valley_paths.png      - static path overlay (README fallback)
  figures/learning_curves.png   - digits loss curves per optimizer (200 steps, lr=0.01, seed 0)
  figures/race_best_loss.png    - best-median-loss bars (from results/results.json)
  figures/market_acc.png        - live-market accuracy bars (from results/results.json)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
import sys
sys.path.insert(0, ROOT)
from src import optimizers as O
from src.valley import loss as vloss, grad as vgrad
from src.data import load_digits_split
from src.mlp import init_params, train, loss_and_grads


def record_valley(name, lr, steps=120):
    opt = O.make_optimizer(name)
    p = [np.array([-10.0, 1.0])]
    st = opt.init_state(p)
    path = [p[0].copy()]
    for t in range(1, steps + 1):
        p = opt.step(p, [vgrad(p[0])], st, lr, t)
        path.append(p[0].copy())
        if not np.all(np.isfinite(p[0])):
            break
    return np.array(path)


def make_gif():
    paths = {"SGD lr=0.019": record_valley("sgd", 0.019),
             "Momentum lr=0.019": record_valley("momentum", 0.019),
             "Adam lr=0.3": record_valley("adam", 0.3)}
    xs = np.linspace(-11, 3, 200)
    ys = np.linspace(-1.5, 1.5, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = 0.5 * X ** 2 + 50 * Y ** 2
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.contour(X, Y, np.log10(Z + 1e-9), levels=14, cmap="terrain", alpha=0.7)
    ax.plot([0], [0], "g*", ms=14, label="minimum (0,0)")
    lines = {}
    dots = {}
    colors = {"SGD lr=0.019": "red", "Momentum lr=0.019": "blue", "Adam lr=0.3": "purple"}
    for k in paths:
        lines[k], = ax.plot([], [], color=colors[k], lw=1.5, label=k)
        dots[k] = ax.plot([], [], "o", color=colors[k], ms=5)[0]
    ax.set_xlim(-11, 3)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("weight 1 (flat direction)")
    ax.set_ylabel("weight 2 (steep direction)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Same valley, same start (-10, 1) — only the update rule differs")
    n = max(len(v) for v in paths.values())

    def upd(f):
        for k, v in paths.items():
            i = min(f, len(v) - 1)
            lines[k].set_data(v[:i + 1, 0], v[:i + 1, 1])
            dots[k].set_data([v[i, 0]], [v[i, 1]])
        return list(lines.values()) + list(dots.values())

    ani = FuncAnimation(fig, upd, frames=list(range(0, n, 2)), interval=80, blit=True)
    out = os.path.join(FIG, "valley_race.gif")
    ani.save(out, writer="pillow", fps=12)
    plt.close(fig)
    # static overlay
    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    ax2.contour(X, Y, np.log10(Z + 1e-9), levels=14, cmap="terrain", alpha=0.7)
    ax2.plot([0], [0], "g*", ms=14)
    for k, v in paths.items():
        ax2.plot(v[:, 0], v[:, 1], color=colors[k], lw=1.5, label=f"{k} ({len(v)} steps)")
    ax2.set_xlim(-11, 3)
    ax2.set_ylim(-1.5, 1.5)
    ax2.legend(fontsize=8)
    ax2.set_title("Valley descent paths — SGD bounces wall-to-wall, Adam glides")
    fig2.savefig(os.path.join(FIG, "valley_paths.png"), dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"valley GIF frames={n // 2}, sizes:",
          os.path.getsize(out) // 1024, "KB")


def make_curves():
    (Xtr, ytr), _ = load_digits_split()
    LAYERS = [64, 32, 32, 32, 32, 32, 10]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name in ["sgd", "momentum", "rmsprop", "adam", "muon"]:
        p0 = init_params(LAYERS, seed=100)
        _, hist = train(p0, O.make_optimizer(name), Xtr, ytr, 0.01, 200, 32, seed=0)
        ax.plot(hist, label=name)
    ax.set_xlabel("step (batch 32)")
    ax.set_ylabel("train loss")
    ax.set_title("Digits training loss — same net, same batches, lr=0.01")
    ax.legend()
    fig.savefig(os.path.join(FIG, "learning_curves.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("learning_curves.png done")


def make_bars():
    with open(os.path.join(ROOT, "results", "results.json")) as f:
        res = json.load(f)
    names = list(res["race"].keys())
    losses = [res["race"][k]["best"]["median_loss"] for k in names]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(names, losses, color=["gray", "skyblue", "orange", "green", "crimson"])
    ax.set_ylabel("best median loss (digits)")
    ax.set_title("Optimizer race — best of grid, median of seeds (lower is better)")
    fig.savefig(os.path.join(FIG, "race_best_loss.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    m = res.get("market_live", {}).get("results", {})
    if m:
        mn = list(m.keys())
        acc = [m[k]["median_acc"] for k in mn]
        fig2, ax2 = plt.subplots(figsize=(7, 4.2))
        ax2.bar(mn, acc, color="teal")
        ax2.axhline(0.5, color="red", ls="--", label="coin flip")
        ax2.set_ylabel("median test accuracy (live BTC direction)")
        ax2.set_title("Live-market transfer — the vision gap collapses (source: "
                      + res["market_live"].get("data_source", "?") + ")")
        ax2.legend()
        fig2.savefig(os.path.join(FIG, "market_acc.png"), dpi=120, bbox_inches="tight")
        plt.close(fig2)
    print("bars done")


if __name__ == "__main__":
    make_gif()
    make_curves()
    make_bars()

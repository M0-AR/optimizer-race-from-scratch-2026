"""SpectraMix v2 verification (post-audit): zero-to-hero re-check.

1. Endpoint trajectory test (multi-step, defeats rank-1 degeneracy).
2. Muon bias-state accumulation test (the fixed silent-drop bug).
3. k re-sweep {8,12,20,30,50} + schedule variant, digits quick grid (5 LRs x 3 seeds x 200).
4. Market GRID (5 LRs x 5 seeds, window=21, live BTC) for adam/muon/smx/smx-sched.
5. Final 400-step x 5-seed confirmation for top-3 at best LRs.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import optimizers as O
from src.data import load_digits_split
from src.mlp import init_params, train, loss_and_grads, accuracy

LAYERS = [64, 32, 32, 32, 32, 32, 10]
LRS = [1e-4, 1e-3, 1e-2, 0.05, 0.2]


def endpoint_trajectory_test():
    rng = np.random.default_rng(7)
    ok = True
    for shape in [(32, 32), (64, 32)]:
        for target, ref_name, kw in [("muon", "muon", {"c0": -10.0}), ("adam", "adam", {"c0": 10.0})]:
            p1 = [rng.standard_normal(shape) * 0.5]
            p2 = [x.copy() for x in p1]
            o_ref, o_smx = O.make_optimizer(ref_name), O.SpectraMix(**kw)
            s_ref, s_smx = o_ref.init_state(p1), o_smx.init_state(p2)
            grng = np.random.default_rng(11)
            for t in range(1, 21):
                g = [grng.standard_normal(shape) * (1 + 0.1 * t)]
                p1 = o_ref.step(p1, g, s_ref, 0.01, t)
                p2 = o_smx.step(p2, g, s_smx, 0.01, t)
            d = float(np.abs(p2[0] - p1[0]).max())
            print(f"shape {shape} 20-step |SMX-{target}|={d:.2e}")
            ok = ok and d < 1e-6
    print("ENDPOINT TRAJECTORIES EXACT" if ok else "ENDPOINT MISMATCH — STOP")
    return ok


def bias_state_test():
    opt = O.make_optimizer("muon")
    p = [np.array([1.0, 2.0])]
    st = opt.init_state(p)
    g = [np.array([0.5, -0.5])]
    p = opt.step(p, g, st, 0.01, 1)
    m1 = st["adam"]["m"][0].copy()
    p = opt.step(p, g, st, 0.01, 2)
    m2 = st["adam"]["m"][0]
    single = (1 - 0.9) * g[0]
    accumulates = not np.allclose(m2, single / (1 - 0.9 ** 2) * 0 + single, atol=1e-12)
    print(f"bias m after 2 steps: {m2} (memory-less would be {single}); accumulates={accumulates}")
    return bool(accumulates and np.all(np.abs(m2) > 0))


def digits_grid(mk, tag):
    (Xtr, ytr), (Xte, yte) = load_digits_split()
    best = None
    for lr in LRS:
        fl, ac = [], []
        for s in [0, 1, 2]:
            p0 = init_params(LAYERS, seed=100 + s)
            p, _ = train(p0, mk(), Xtr, ytr, lr, 200, 32, seed=s)
            fl.append(float(loss_and_grads(p, Xtr, ytr)[0]))
            ac.append(float(accuracy(p, Xte, yte)))
        row = (lr, sorted(fl)[1], sorted(ac)[1])
        if best is None or row[1] < best[1]:
            best = row
    print(f"[digits] {tag:22s} lr={best[0]:<7g} loss={best[1]:.4f} acc={best[2]:.3f}")
    return best


def market_grid():
    import src.market_live as ML
    from src.mlp import init_params as ip, train as tr, accuracy as ac
    LAY, MLRS = [21, 16, 8, 2], [1e-3, 3e-3, 1e-2, 3e-2, 0.1]
    try:
        px, src = ML.fetch_btc_daily()
    except Exception as e:
        px, src = ML.synthetic_market()
        src = f"synthetic ({e})"
    X, y = ML.build_features(px, window=21)
    n, split = len(X), int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    mu, sd = Xtr.mean(), Xtr.std() + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    print(f"[market {src}] n_train={split} n_test={n - split}")
    out = {}
    for name, mk in [("adam", lambda: O.make_optimizer("adam")),
                     ("muon", lambda: O.make_optimizer("muon")),
                     ("spectramix", lambda: O.make_optimizer("spectramix")),
                     ("smx-sched", lambda: O.make_optimizer("spectramix", schedule=True))]:
        best = None
        for lr in MLRS:
            acs = []
            for s in range(5):
                p0 = ip(LAY, seed=200 + s)
                p, _ = tr(p0, mk(), Xtr, ytr, lr, 200, 32, seed=s)
                acs.append(float(ac(p, Xte, yte)))
            row = (lr, sorted(acs)[2], sorted(acs))
            if best is None or row[1] > best[1]:
                best = row
        out[name] = best
        print(f"[market] {name:10s} lr={best[0]:<7g} median acc={best[1]:.3f} spread={best[2]}")
    return out


def final_400():
    (Xtr, ytr), (Xte, yte) = load_digits_split()
    for name, mk, lr in [("muon", lambda: O.make_optimizer("muon"), 0.01),
                         ("spectramix", lambda: O.make_optimizer("spectramix"), 0.01),
                         ("adam", lambda: O.make_optimizer("adam"), 0.01)]:
        fl, ac = [], []
        for s in range(5):
            p0 = init_params(LAYERS, seed=100 + s)
            p, _ = train(p0, mk(), Xtr, ytr, lr, 400, 32, seed=s)
            fl.append(float(loss_and_grads(p, Xtr, ytr)[0]))
            ac.append(float(accuracy(p, Xte, yte)))
        print(f"[400-step] {name:10s} loss={sorted(fl)[2]:.4f} acc={sorted(ac)[2]:.3f}")


if __name__ == "__main__":
    assert endpoint_trajectory_test(), "endpoints broken"
    assert bias_state_test(), "bias state broken"
    for k in [8, 12, 20, 30, 50]:
        digits_grid(lambda k=k: O.SpectraMix(k=k), f"spectramix k={k}")
    digits_grid(lambda: O.make_optimizer("muon"), "muon (fixed)")
    digits_grid(lambda: O.make_optimizer("spectramix", schedule=True), "smx-sched")
    market_grid()
    final_400()

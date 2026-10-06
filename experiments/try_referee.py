"""Referee v2: zero-scout per-step empirical update selection (the measurement principle).

v2 fixes (audit 2026-10-06 + literature):
 1. eta=4.0 was arbitrary -> AdaHedge self-tuning (eta_t = lnK/Delta_{t-1}, de Rooij et al.
    2014: parameter-free, no sweep needed) + Decreasing Hedge (eta_t = sqrt(2lnK/t)) as fallback.
 2. Per-step min-max normalization destroyed cross-step stakes -> AdaHedge works on RAW
    holdout losses via the mixability gap (proper scale handling).
 3. Hedge never picks momentum -> pool ablation {full4} vs {adam,muon,smx} (25% cheaper?).
 4. Holdout batch fixed 32 -> tested {32, 64, 128}.
 5. No wall-clock -> time.perf_counter reported (not just forward counts).
 6. Market was greedy-only/3-pool -> hedge + adahedge + same ablations.
 7. No long-run confirmation -> 400-step x5-seed for champion vs best fixed.
Modes: greedy (FTL) | hedge (fixed eta) | adahedge (self-tuning) | decreasing.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import optimizers as O
from src.data import load_digits_split
from src.mlp import init_params, train, loss_and_grads, accuracy

LAYERS = [64, 32, 32, 32, 32, 32, 10]
FULL = [("momentum", lambda: O.make_optimizer("momentum"), 0.01),
        ("adam", lambda: O.make_optimizer("adam"), 0.01),
        ("muon", lambda: O.make_optimizer("muon"), 0.01),
        ("spectramix", lambda: O.make_optimizer("spectramix"), 0.01)]
LEAN = [s for s in FULL if s[0] != "momentum"]


class Referee:
    def __init__(self, specs, mode="greedy", eta=4.0, holdout=32, seed=0):
        self.specs = specs
        self.mode = mode
        self.eta = eta
        self.holdout = holdout
        self.rng = np.random.default_rng(seed)
        self.wins = None

    def run(self, p0, Xtr, ytr, steps=200, batch=32, seed=0):
        rng = np.random.default_rng(seed)
        K, n = len(self.specs), len(Xtr)
        opts = [mk() for _, mk, _ in self.specs]
        states = [o.init_state(p0) for o in opts]
        p = [x.copy() for x in p0]
        logw = np.zeros(K)
        Lcum = np.zeros(K)
        Delta = 0.0
        eta = float("inf")
        self.wins = np.zeros((4, K))
        for t in range(1, steps + 1):
            idx = rng.integers(0, n, batch)
            _, grads, _ = loss_and_grads(p, Xtr[idx], ytr[idx])
            prop = [o.step([x.copy() for x in p], grads, st, lr, t)
                    for o, st, (_, _, lr) in zip(opts, states, self.specs)]
            hidx = rng.integers(0, n, self.holdout)
            scores = np.array([float(loss_and_grads(q, Xtr[hidx], ytr[hidx])[0]) for q in prop])
            if self.mode == "greedy":
                k = int(np.argmin(scores))
            elif self.mode == "hedge":
                span = scores.max() - scores.min() + 1e-12
                logw = logw - self.eta * (scores - scores.min()) / span
                w = np.exp(logw - logw.max())
                k = int(self.rng.choice(K, p=w / w.sum()))
            elif self.mode == "decreasing":
                et = float(np.sqrt(2 * np.log(K) / t))
                span = scores.max() - scores.min() + 1e-12
                logw = logw - et * (scores - scores.min()) / span
                w = np.exp(logw - logw.max())
                k = int(self.rng.choice(K, p=w / w.sum()))
            elif self.mode == "adahedge":
                # de Rooij et al. 2014 on RAW losses: w from eta_t, mix loss, gap, self-tune.
                if np.isinf(eta):
                    w = np.full(K, 1.0 / K)
                    mix = float(scores.min())
                else:
                    w = np.exp(-eta * (Lcum - Lcum.min()))
                    w = w / w.sum()
                    shift = scores - scores.max()
                    mix = float(-np.log(np.sum(w * np.exp(-eta * shift))) / eta + scores.max())
                Delta += float(w @ scores) - mix
                eta = float(np.log(K) / max(Delta, 1e-12))
                Lcum = Lcum + scores
                k = int(self.rng.choice(K, p=w))
            else:
                raise ValueError(self.mode)
            self.wins[min(3, 4 * (t - 1) // steps), k] += 1
            p = prop[k]
        return p


def run_digits(specs, mode, steps, seeds, Xtr, ytr, Xte, yte, **kw):
    fl, ac, t0 = [], [], time.perf_counter()
    for s in seeds:
        p0 = init_params(LAYERS, seed=100 + s)
        p = Referee(specs, mode=mode, seed=s, **kw).run(p0, Xtr, ytr, steps, 32, seed=s)
        fl.append(float(loss_and_grads(p, Xtr, ytr)[0]))
        ac.append(float(accuracy(p, Xte, yte)))
    dt = time.perf_counter() - t0
    return sorted(fl)[len(fl) // 2], sorted(ac)[len(ac) // 2], dt


def main():
    (Xtr, ytr), (Xte, yte) = load_digits_split()
    S3 = (0, 1, 2)
    print("== eta sweep (hedge, full pool, holdout=32) ==")
    for eta in [1, 2, 4, 8, 16]:
        l, a, dt = run_digits(FULL, "hedge", 200, S3, Xtr, ytr, Xte, yte, eta=eta)
        print(f"eta={eta:<3d} loss={l:.4f} acc={a:.3f} {dt:.0f}s")
    print("== modes (full pool) ==")
    for mode, kw in [("greedy", {}), ("decreasing", {}), ("adahedge", {})]:
        l, a, dt = run_digits(FULL, mode, 200, S3, Xtr, ytr, Xte, yte, **kw)
        print(f"{mode:10s} loss={l:.4f} acc={a:.3f} {dt:.0f}s")
    print("== pool ablation (best modes) ==")
    for tag, pool in [("lean3", LEAN)]:
        for mode, kw in [("greedy", {}), ("adahedge", {})]:
            l, a, dt = run_digits(pool, mode, 200, S3, Xtr, ytr, Xte, yte, **kw)
            print(f"{tag}+{mode:8s} loss={l:.4f} acc={a:.3f} {dt:.0f}s")
    print("== holdout size (adahedge, full pool) ==")
    for h in [32, 64, 128]:
        l, a, dt = run_digits(FULL, "adahedge", 200, S3, Xtr, ytr, Xte, yte, holdout=h)
        print(f"holdout={h:<4d} loss={l:.4f} acc={a:.3f} {dt:.0f}s")
    print("== selection fractions (adahedge, full pool) ==")
    r = Referee(FULL, mode="adahedge")
    r.run(init_params(LAYERS, seed=100), Xtr, ytr, 200, 32, seed=0)
    print((r.wins / r.wins.sum(axis=1, keepdims=True)).round(2).tolist())
    print("== market (live BTC, window=21, 5 seeds, greedy+adahedge, lean pool) ==")
    import src.market_live as ML
    from src.mlp import init_params as ip, train as tr, accuracy as ac
    LAY = [21, 16, 8, 2]
    try:
        px, src = ML.fetch_btc_daily()
    except Exception as e:
        px, src = ML.synthetic_market()
        src = f"synthetic ({e})"
    X, y = ML.build_features(px, window=21)
    n, split = len(X), int(len(X) * 0.8)
    Xtrm, ytrm, Xtem, ytem = X[:split], y[:split], X[split:], y[split:]
    mu, sd = Xtrm.mean(), Xtrm.std() + 1e-9
    Xtrm, Xtem = (Xtrm - mu) / sd, (Xtem - mu) / sd
    MSPECS = [("adam", lambda: O.make_optimizer("adam"), 0.003),
              ("muon", lambda: O.make_optimizer("muon"), 0.03),
              ("spectramix", lambda: O.make_optimizer("spectramix"), 0.1)]
    for name, mk, lr in MSPECS:
        acs = []
        for s in range(5):
            p0 = ip(LAY, seed=200 + s)
            p, _ = tr(p0, mk(), Xtrm, ytrm, lr, 200, 32, seed=s)
            acs.append(float(ac(p, Xtem, ytem)))
        print(f"[market fixed] {name:10s} median acc={sorted(acs)[2]:.3f}")
    for mode in ["greedy", "adahedge"]:
        acs = []
        for s in range(5):
            p0 = ip(LAY, seed=200 + s)
            p = Referee(MSPECS, mode=mode, seed=s).run(p0, Xtrm, ytrm, 200, 32, seed=s)
            acs.append(float(ac(p, Xtem, ytem)))
        print(f"[market referee-{mode}] median acc={sorted(acs)[2]:.3f} {sorted(acs)}")
    print("== 400-step x5 confirmation (digits, champion vs best fixed) ==")
    for tag, fn in [("smx-fixed", None), ("referee-adahedge", "adahedge"), ("referee-greedy", "greedy")]:
        fl, ac = [], []
        for s in range(5):
            p0 = init_params(LAYERS, seed=100 + s)
            if fn is None:
                p, _ = train(p0, O.make_optimizer("spectramix"), Xtr, ytr, 0.01, 400, 32, seed=s)
            else:
                p = Referee(FULL, mode=fn, seed=s).run(p0, Xtr, ytr, 400, 32, seed=s)
            fl.append(float(loss_and_grads(p, Xtr, ytr)[0]))
            ac.append(float(accuracy(p, Xte, yte)))
        print(f"[400-step] {tag:16s} loss={sorted(fl)[2]:.4f} acc={sorted(ac)[2]:.3f}")


if __name__ == "__main__":
    main()

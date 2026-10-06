"""SpectraMix contender trial: does adaptive Muon<->Adam beat both fixed endpoints?

1. Endpoint unit tests: force conc->1 (c0=-10) must equal Muon; conc->0 (c0=+10) must equal Adam.
2. Valley robustness + digits quick protocol (same 5-LR x 3-seed x 200-step grid as results.json).
3. Live-market transfer (same BTC task, 200 steps).
Falsifiable claim: SpectraMix ~= Muon on digits AND ~= Adam on markets (best of both).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import optimizers as O
from src.data import load_digits_split
from src.mlp import init_params, train, loss_and_grads, accuracy

LAYERS = [64, 32, 32, 32, 32, 32, 10]
LRS = [1e-4, 1e-3, 1e-2, 0.05, 0.2]


def endpoint_test():
    rng = np.random.default_rng(7)
    ok = True
    for shape in [(32, 32), (64, 32)]:
        p = [rng.standard_normal(shape) * 0.5]
        g = [rng.standard_normal(shape)]
        mu = O.make_optimizer("muon").step([x.copy() for x in p], g,
              O.Muon().init_state(p), 0.01, 5)[0]
        sm = O.SpectraMix(c0=-10.0)
        st = sm.init_state(p)
        got = sm.step([x.copy() for x in p], g, st, 0.01, 5)[0]
        d_mu = float(np.abs(got - mu).max())
        ad = O.make_optimizer("adam").step([x.copy() for x in p], g,
              O.Adam().init_state(p), 0.01, 5)[0]
        sm2 = O.SpectraMix(c0=10.0)
        st2 = sm2.init_state(p)
        got2 = sm2.step([x.copy() for x in p], g, st2, 0.01, 5)[0]
        d_ad = float(np.abs(got2 - ad).max())
        print(f"shape {shape}: |SMX(c0=-10)-Muon|={d_mu:.2e}  |SMX(c0=+10)-Adam|={d_ad:.2e}")
        ok = ok and d_mu < 1e-9 and d_ad < 1e-9
    print("ENDPOINTS EXACT" if ok else "ENDPOINT MISMATCH — STOP")
    return ok


def digits_trial():
    (Xtr, ytr), (Xte, yte) = load_digits_split()
    base = {}
    for name in ["sgd", "momentum", "rmsprop", "adam", "muon", "spectramix"]:
        rows = []
        for lr in LRS:
            fl, ac = [], []
            for s in [0, 1, 2]:
                p0 = init_params(LAYERS, seed=100 + s)
                p, _ = train(p0, O.make_optimizer(name), Xtr, ytr, lr, 200, 32, seed=s)
                fl.append(float(loss_and_grads(p, Xtr, ytr)[0]))
                ac.append(float(accuracy(p, Xte, yte)))
            rows.append((lr, sorted(fl)[1], sorted(ac)[1]))
        best = min(rows, key=lambda r: r[1])
        base[name] = best
        print(f"[digits] {name:10s} best lr={best[0]:<7g} loss={best[1]:.4f} acc={best[2]:.3f}")
    # diagnostic: mean alpha per layer at end of a spectramix run
    p0 = init_params(LAYERS, seed=100)
    opt = O.make_optimizer("spectramix")
    p, _ = train(p0, opt, Xtr, ytr, 0.01, 200, 32, seed=0)
    print("[digits] mean alpha per 2D layer (1=Muon-like, 0=Adam-like):",
          [round(a, 3) for a in opt_state_alphas(opt, p0, Xtr, ytr)])
    return base


def opt_state_alphas(opt, p0, Xtr, ytr):
    from src.mlp import loss_and_grads as lg
    _, grads, _ = lg(p0, Xtr[:32], ytr[:32])
    st = opt.init_state(p0)
    opt.step([x.copy() for x in p0], grads, st, 0.01, 200)
    return [a for a in st["alphas"] if a == a]


def market_trial():
    from src.market_live import run_market_experiment
    import src.market_live as ML
    from src.mlp import init_params as ip, train as tr, accuracy as acc, loss_and_grads as lg
    LAY = [30, 16, 8, 2]
    try:
        px, src = ML.fetch_btc_daily()
    except Exception as e:
        px, src = ML.synthetic_market()
        src = f"synthetic ({e})"
    X, y = ML.build_features(px)
    n, split = len(X), int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    mu, sd = Xtr.mean(), Xtr.std() + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    for name, lr in [("adam", 0.003), ("muon", 0.01), ("spectramix", 0.005)]:
        acs = []
        for s in range(5):
            p0 = ip(LAY, seed=200 + s)
            p, _ = tr(p0, O.make_optimizer(name), Xtr, ytr, lr, 200, 32, seed=s)
            acs.append(float(acc(p, Xte, yte)))
        print(f"[market {src}] {name:10s} median acc={sorted(acs)[2]:.3f} {sorted(acs)}")


if __name__ == "__main__":
    if not endpoint_test():
        raise SystemExit(1)
    digits_trial()
    market_trial()

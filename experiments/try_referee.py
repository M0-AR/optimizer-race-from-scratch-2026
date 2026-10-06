"""Referee: zero-scout per-step empirical update selection (the measurement principle).

PRINCIPLE: don't MODEL the landscape (moments, spectra, curvature proxies) to choose an
update — MEASURE candidate updates directly on fresh data and take the winner. ROR (2026)
showed shorter scouts dominate exhaustive search; the limit s->0 (no scout training at all,
just forward evaluation) was unoccupied: epoch-level selection is taken (ROR/AOS/OptiRoulette/
RL-choose, all 2026), per-step zero-scout selection is not. This also answers DynMuon's open
problem (select spectral shaping online by observed statistics) by direct measurement instead
of a fixed time schedule.

PROTOCOL (shared-params experts, full information -> Hedge framing):
  candidates k=1..K (each: own optimizer + own best LR, own persistent state).
  each step: train batch -> true grad g; every candidate proposes next params from the SAME
  current params (states all observe g); score each proposal with ONE forward pass on a FRESH
  holdout batch (never trained on); commit argmin (greedy) or Hedge-sample; all states advance
  with g regardless (experts observe everything).
COST: K forward passes ~= +1 step (forward ~= 1/3 of fwd+bwd). Counted and reported.
DIAGNOSTIC (itself a finding): selection fractions per quarter — does the loss itself prefer
Muon early and Adam late? That would rediscover DynMuon's schedule from data.
CAVEAT (ROR authors' own): reusing one holdout for selection each step can overfit the
holdout; mitigated by fresh random batches (streaming holdout, not a fixed set).
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import optimizers as O
from src.data import load_digits_split
from src.mlp import init_params, loss_and_grads, accuracy

LAYERS = [64, 32, 32, 32, 32, 32, 10]


class Referee:
    def __init__(self, specs, mode="greedy", eta=4.0, seed=0):
        """specs: [(name, mk_fn, lr)]. mode: greedy | hedge."""
        self.specs = specs
        self.mode = mode
        self.eta = eta
        self.rng = np.random.default_rng(seed)
        self.wins = None

    def run(self, p0, Xtr, ytr, steps=200, batch=32, seed=0):
        rng = np.random.default_rng(seed)
        n = len(Xtr)
        opts = [mk() for _, mk, _ in self.specs]
        states = [o.init_state(p0) for o in opts]
        p = [x.copy() for x in p0]
        logw = np.zeros(len(opts))
        self.wins = np.zeros((4, len(opts)))
        fwds = 0
        for t in range(1, steps + 1):
            idx = rng.integers(0, n, batch)
            _, grads, _ = loss_and_grads(p, Xtr[idx], ytr[idx])
            prop = [o.step([x.copy() for x in p], grads, st, lr, t)
                    for o, st, (_, _, lr) in zip(opts, states, self.specs)]
            hidx = rng.integers(0, n, batch)  # fresh holdout micro-batch
            scores = []
            for q in prop:
                l, _, _ = loss_and_grads(q, Xtr[hidx], ytr[hidx])
                scores.append(float(l))
            fwds += len(prop)
            scores = np.array(scores)
            if self.mode == "greedy":
                k = int(np.argmin(scores))
            else:  # hedge over normalized advantage (bounded in [0,1])
                rng_ = self.rng
                span = scores.max() - scores.min() + 1e-12
                lnorm = (scores - scores.min()) / span
                logw = logw - self.eta * lnorm
                w = np.exp(logw - logw.max())
                w = w / w.sum()
                k = int(rng_.choice(len(w), p=w))
            self.wins[min(3, 4 * (t - 1) // steps), k] += 1
            p = prop[k]
        return p, fwds


def digits_trial(steps=200, seeds=(0, 1, 2)):
    (Xtr, ytr), (Xte, yte) = load_digits_split()
    specs = [("momentum", lambda: O.make_optimizer("momentum"), 0.01),
             ("adam", lambda: O.make_optimizer("adam"), 0.01),
             ("muon", lambda: O.make_optimizer("muon"), 0.01),
             ("spectramix", lambda: O.make_optimizer("spectramix"), 0.01)]
    from src.mlp import train
    base = {}
    for name, mk, lr in specs:  # fixed-candidate baselines, same seeds/steps
        fl, ac = [], []
        for s in seeds:
            p0 = init_params(LAYERS, seed=100 + s)
            p, _ = train(p0, mk(), Xtr, ytr, lr, steps, 32, seed=s)
            fl.append(float(loss_and_grads(p, Xtr, ytr)[0]))
            ac.append(float(accuracy(p, Xte, yte)))
        base[name] = (sorted(fl)[1], sorted(ac)[1])
        print(f"[digits fixed] {name:10s} loss={base[name][0]:.4f} acc={base[name][1]:.3f}")
    for mode in ["greedy", "hedge"]:
        fl, ac, fw = [], [], 0
        for s in seeds:
            p0 = init_params(LAYERS, seed=100 + s)
            t0 = time.time()
            p, f = Referee(specs, mode=mode, seed=s).run(p0, Xtr, ytr, steps, 32, seed=s)
            fw += f
            fl.append(float(loss_and_grads(p, Xtr, ytr)[0]))
            ac.append(float(accuracy(p, Xte, yte)))
        r = Referee(specs, mode=mode)
        p0 = init_params(LAYERS, seed=100)
        r.run(p0, Xtr, ytr, steps, 32, seed=0)
        frac = (r.wins / r.wins.sum(axis=1, keepdims=True)).round(2)
        print(f"[digits referee-{mode}] loss={sorted(fl)[1]:.4f} acc={sorted(ac)[1]:.3f} "
              f"extra-forwards={fw} (~{fw / (len(seeds) * steps * 3):.1f}x step)")
        print(f"  selection fractions per quarter [momentum,adam,muon,smx]: {frac.tolist()}")
    return base


def market_trial():
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
    Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
    mu, sd = Xtr.mean(), Xtr.std() + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    print(f"[market {src}] n_train={split} n_test={n - split}")
    specs = [("adam", lambda: O.make_optimizer("adam"), 0.003),
             ("muon", lambda: O.make_optimizer("muon"), 0.03),
             ("spectramix", lambda: O.make_optimizer("spectramix"), 0.1)]
    for name, mk, lr in specs:
        acs = []
        for s in range(5):
            p0 = ip(LAY, seed=200 + s)
            p, _ = tr(p0, mk(), Xtr, ytr, lr, 200, 32, seed=s)
            acs.append(float(ac(p, Xte, yte)))
        print(f"[market fixed] {name:10s} median acc={sorted(acs)[2]:.3f}")
    acs = []
    for s in range(5):
        p0 = ip(LAY, seed=200 + s)
        p, _ = Referee(specs, mode="greedy", seed=s).run(p0, Xtr, ytr, 200, 32, seed=s)
        acs.append(float(ac(p, Xte, yte)))
    print(f"[market referee-greedy] median acc={sorted(acs)[2]:.3f} {sorted(acs)}")


if __name__ == "__main__":
    digits_trial()
    market_trial()

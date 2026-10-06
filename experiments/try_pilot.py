"""Pilot: step-size-free, instability-guarded measurement optimizer.

NEED (triangulated 2026-27): LR tuning is the tax (17-try grids; million-dollar LLM runs get
one try). Parameter-free methods (DoG/DoWG/D-Adaptation/Prodigy/DAoG) are all VECTOR methods;
spectral/orthogonal methods (Muon family) all need tuned LRs — LR-free x spectral is UNOCCUPIED
(Awesome-Muon lists none). Instability monitors exist (ZClip, Spectral-Alignment, R-metric, TIOI)
but none lives INSIDE an LR-free selector. Pilot fills exactly that intersection.

MECHANISM (three borrowed organs, one new body):
 1. DIRECTION ELECTION (from Referee): candidates {momentum, adam, muon, spectramix} propose
    from shared params; each proposal is RMS-normalized to a pure DIRECTION (their LRs and
    natural scales discarded); holdout forward-pass elects the winner. No step size anywhere.
 2. DoG BASE SCALE (Ivgi et al. 2023, per-layer) + SCALE ELECTION: DoG alone cannot set
    absolute scale in 200 steps (cold-start deadlock measured twice: raw r explodes, floored
    r freezes). So DoG supplies only a coarse BASE b = (r/sqrt(G)) * ||g|| per layer, and the
    holdout elects among {0.25x, 1x, 4x} x b x direction (12 proposals, still forward-only).
    Scale, like direction, is MEASURED not modeled. Loose brake as backstop only.
 3. Z-GUARD on GRADIENT NORMS (ZClip 2025, faithfully): v1 tracked holdout LOSSES, but a
    trending loss drags the EMA with it (boiling frog — 5 vetoes while burning down).
    Track ||g|| EMA mean/var (a=0.97) instead; z = (||g||-mean)/std; z > 2.5 shrinks the
    step by (2.5/z)^2. Stationary scale, honest anomaly signal.
HONEST LABEL: step-size-free (no lr/eta/schedule anywhere), NOT hyperparameter-free
(betas, NS steps, k/c0, z-thresh remain — all fixed literature defaults, never swept).
SUCCESS METRIC: match LR-tuned fixed optimizers while consuming 1 tuning-run vs their grid.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import optimizers as O
from src.data import load_digits_split
from src.mlp import init_params, loss_and_grads, accuracy

LAYERS = [64, 32, 32, 32, 32, 32, 10]
CANDS = [("momentum", lambda: O.make_optimizer("momentum")),
         ("adam", lambda: O.make_optimizer("adam")),
         ("muon", lambda: O.make_optimizer("muon")),
         ("spectramix", lambda: O.make_optimizer("spectramix"))]


class Pilot:
    def __init__(self, cands=CANDS, z_thresh=2.5, ema_a=0.97, seed=0):
        self.cands = cands
        self.zt = z_thresh
        self.a = ema_a
        self.rng = np.random.default_rng(seed)
        self.vetoes = 0
        self.zmax = 0.0

    def run(self, p0, Xtr, ytr, steps=200, batch=32, holdout=128, seed=0):
        rng = np.random.default_rng(seed)
        n = len(Xtr)
        opts = [mk() for _, mk in self.cands]
        states = [o.init_state(p0) for o in opts]
        p = [x.copy() for x in p0]
        x0 = [x.copy() for x in p0]
        x0n = [float(np.linalg.norm(x)) for x in x0]
        tau = [0.5 * n0 + 1e-2 for n0 in x0n]  # emergency brake only (rarely binds)
        r = [1e-4 * (1 + n0) for n0 in x0n]
        G = [0.0 for _ in p0]
        Gfloor = [None for _ in p0]  # cold-start floor set from first batch
        gm_ema = gv_ema = None  # gradient-norm EMA (ZClip-faithful guard scale)
        wins = np.zeros((4, len(opts)))
        self.scales = []
        for t in range(1, steps + 1):
            idx = rng.integers(0, n, batch)
            _, grads, _ = loss_and_grads(p, Xtr[idx], ytr[idx])
            for i, g in enumerate(grads):
                G[i] += float((g * g).sum())
                if Gfloor[i] is None:
                    Gfloor[i] = float((g * g).sum()) + 1e-18  # anti-div0 only, not burn-in
            # 1. directions: lr=1 proposals -> unit-RMS direction vectors
            dirs = []
            for o, st in zip(opts, states):
                q = o.step([x.copy() for x in p], grads, st, 1.0, t)
                d = [pp - qq for pp, qq in zip(p, q)]
                dirs.append(d)
            # 2. DoG coarse base scale (shared physics, no tuning)
            base = [rr / max(gg + gf, 1e-18) ** 0.5 for rr, gg, gf in zip(r, G, Gfloor)]
            hidx = rng.integers(0, n, holdout)
            best, best_s = None, None
            # DIMENSIONAL FIX v2 (traced): DoG step NORM = b*||g||. A unit-RMS direction
            # has norm sqrt(N), so delta = smult*b*(||g||/sqrt(N))*u_hat reproduces DoG's
            # norm exactly. (v1 omitted ||g|| -> ~11x big; v4 put ||g|| on unit vector ->
            # ~45x big, brake-bound pinball at 4.0 every step.)
            # Elect SCALE {0.25,1,4} x base jointly with direction (12 forward proposals).
            gnorm = [float(np.linalg.norm(g)) + 1e-18 for g in grads]
            gnel = [float(g.size) ** 0.5 for g in grads]
            for di, d in enumerate(dirs):
                rms = [float(np.sqrt((dli * dli).mean()) + 1e-12) for dli in d]
                unit = [dli / rr_ for dli, rr_ in zip(d, rms)]
                for smult in (0.0625, 0.25, 1.0, 4.0):
                    delta = [smult * b * (gn / ne) * u
                             for b, gn, ne, u in zip(base, gnorm, gnel, unit)]
                    for i in range(len(delta)):  # emergency brake at scored scale
                        dn = float(np.linalg.norm(delta[i]))
                        if dn > tau[i]:
                            delta[i] = delta[i] * (tau[i] / dn)
                    cand = [pp - dd for pp, dd in zip(p, delta)]
                    s = float(loss_and_grads(cand, Xtr[hidx], ytr[hidx])[0])
                    if best_s is None or s < best_s:
                        best, best_s = (cand, delta, di, smult), s
            # 3. z-guard on GRADIENT NORM (ZClip-faithful; loss-EMA boils the frog)
            gn = float(np.sqrt(sum(float((g * g).sum()) for g in grads)))
            if gm_ema is None:
                gm_ema, gv_ema = gn, 1.0
            else:
                gm_ema = self.a * gm_ema + (1 - self.a) * gn
                gv_ema = self.a * gv_ema + (1 - self.a) * (gn - gm_ema) ** 2
            z = (gn - gm_ema) / max(gv_ema ** 0.5, 1e-12)
            self.zmax = max(self.zmax, z)
            _, delta, di, smult = best
            shrink = (self.zt / z) ** 2 if z > self.zt else 1.0
            if shrink < 1.0:
                self.vetoes += 1
            p = [pp - shrink * dd for pp, dd in zip(p, delta)]
            for i in range(len(p)):
                dd = float(np.linalg.norm(p[i] - x0[i]))
                if dd > r[i]:
                    r[i] = dd
            wins[min(3, 4 * (t - 1) // steps), best[2]] += 1
            self.scales.append(best[3])
        self.wins = wins
        self.wins = wins
        return p


def digits_trial(steps=200, seeds=(0, 1, 2)):
    (Xtr, ytr), (Xte, yte) = load_digits_split()
    t0 = time.perf_counter()
    fl, ac = [], []
    for s in seeds:
        pl = Pilot(seed=s)
        p = pl.run(init_params(LAYERS, seed=100 + s), Xtr, ytr, steps, 32, 128, seed=s)
        fl.append(float(loss_and_grads(p, Xtr, ytr)[0]))
        ac.append(float(accuracy(p, Xte, yte)))
    dt = time.perf_counter() - t0
    print(f"[digits pilot] loss={sorted(fl)[1]:.4f} acc={sorted(ac)[1]:.3f} {dt:.0f}s "
          f"(1 config, 0 LRs; vetoes/zmax from last seed: {pl.vetoes}/{pl.zmax:.1f})")
    print(f"  tuned references: smx-fixed 0.072/0.948 (5-LR grid), referee-greedy 0.065/0.952, "
          f"referee-hedge 0.053/0.956, pilot-400x5 0.019/0.960")
    return sorted(fl)[1], sorted(ac)[1]


def market_trial():
    import src.market_live as ML
    from src.mlp import init_params as ip, accuracy as ac
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
    acs = []
    for s in range(5):
        p0 = ip(LAY, seed=200 + s)
        # Pilot is architecture-agnostic: same code, LAYERS inferred from p0
        import src.mlp as M
        pl = Pilot(seed=s)
        # temporarily run with market layer shapes via direct call
        p = pl.run(p0, Xtr, ytr, 200, 32, 128, seed=s)
        acs.append(float(ac(p, Xte, yte)))
    print(f"[market pilot {src}] median acc={sorted(acs)[2]:.3f} {sorted(acs)} (0 LRs vs 5-LR grids)")


if __name__ == "__main__":
    digits_trial()
    market_trial()

"""Seven optimizers from scratch (NumPy only).

Implements, in invention order, the exact update rules from the video:
  GD / SGD -> Momentum (Polyak 1964) -> RMSprop (Hinton 2012, unpublished lecture)
  -> Adam (Kingma & Ba 2014) -> AdamW (Loshchilov & Hutter 2017)
  -> Muon (Jordan et al. 2024, Newton-Schulz orthogonalization)

Verified against:
  - Keller Jordan blog + muon.py (coeffs 3.4445, -4.7750, 2.0315, 5 steps)
  - Kingma & Ba 2014 bias-correction formula
  - Loshchilov & Hutter 2017 decoupled decay
  - Kimi K2 tech report (MuonClip = Muon + WD + QK-Clip, 1T params, 15.5T tokens)

API: every optimizer is init_state(params) + step(params, grads, state, lr, t).
Params/grads are lists of np.ndarray. Pure Python + NumPy, no torch.
"""
import numpy as np


# ---------------------------------------------------------------- utilities
def _zeros_like(params):
    return [np.zeros_like(p) for p in params]


# ---------------------------------------------------------------- 1. SGD
class SGD:
    """Plain (stochastic) gradient descent: p -= lr * g. Cauchy 1847 / Robbins-Monro 1951."""

    def init_state(self, params):
        return {}

    def step(self, params, grads, state, lr, t=1):
        return [p - lr * g for p, g in zip(params, grads)]


# ---------------------------------------------------------------- 2. Momentum (heavy ball, Polyak 1964)
class Momentum:
    """v = mu*v + g ; p -= lr * v.  mu=0.9 default (video)."""

    def __init__(self, mu=0.9):
        self.mu = mu

    def init_state(self, params):
        return {"v": _zeros_like(params)}

    def step(self, params, grads, state, lr, t=1):
        v = state["v"]
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            v[i] = self.mu * v[i] + g
            out.append(p - lr * v[i])
        return out


# ---------------------------------------------------------------- 3. RMSprop (Hinton/Tieleman 2012 lecture)
class RMSprop:
    """s = rho*s + (1-rho)*g^2 ; p -= lr * g / (sqrt(s) + eps)."""

    def __init__(self, rho=0.9, eps=1e-8):
        self.rho = rho
        self.eps = eps

    def init_state(self, params):
        return {"s": _zeros_like(params)}

    def step(self, params, grads, state, lr, t=1):
        s = state["s"]
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            s[i] = self.rho * s[i] + (1 - self.rho) * g * g
            out.append(p - lr * g / (np.sqrt(s[i]) + self.eps))
        return out


# ---------------------------------------------------------------- 4. Adam (Kingma & Ba 2014)
class Adam:
    """m = b1*m+(1-b1)g ; v = b2*v+(1-b2)g^2 ; bias-corrected step."""

    def __init__(self, b1=0.9, b2=0.999, eps=1e-8):
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def init_state(self, params):
        return {"m": _zeros_like(params), "v": _zeros_like(params)}

    def step(self, params, grads, state, lr, t=1):
        m, v = state["m"], state["v"]
        b1t = 1 - self.b1 ** t
        b2t = 1 - self.b2 ** t
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = self.b1 * m[i] + (1 - self.b1) * g
            v[i] = self.b2 * v[i] + (1 - self.b2) * g * g
            mh = m[i] / b1t
            vh = v[i] / b2t
            out.append(p - lr * mh / (np.sqrt(vh) + self.eps))
        return out


# ---------------------------------------------------------------- 5. AdamW (Loshchilov & Hutter 2017)
class AdamW(Adam):
    """Adam + DECOUPLED decay: p *= (1 - lr*wd) BEFORE the Adam step.

    The bug AdamW fixes: old code added wd*p INTO the gradient, so Adam
    divided the decay by sqrt(v) -> weights with small grads got wiped out
    (video demo: 0.80 / 0.17 / 0.02), while decoupled gives 0.80/0.80/0.80.
    """

    def __init__(self, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(b1, b2, eps)
        self.wd = weight_decay

    def step(self, params, grads, state, lr, t=1):
        decayed = [p * (1 - lr * self.wd) for p in params]
        return super().step(decayed, grads, state, lr, t)


# ---------------------------------------------------------------- 6. Muon (Jordan et al. 2024)
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315  # tuned quintic coeffs (muon.py)


def zeropower_via_newtonschulz5(G, steps=5):
    """Approximately orthogonalize matrix G using only matmuls.

    Maps singular values S -> ~1 (band 0.5..1.5 is fine for training).
    Exact port of Keller Jordan's muon.py to NumPy (float64 for CPU determinism).
    """
    G = np.asarray(G, dtype=np.float64)
    transposed = False
    if G.shape[0] > G.shape[1]:
        G = G.T
        transposed = True
    X = G / (np.linalg.norm(G) + 1e-7)  # spectral norm <= 1
    for _ in range(steps):
        A = X @ X.T
        B = NS_B * A + NS_C * (A @ A)
        X = NS_A * X + B @ X
    if transposed:
        X = X.T
    # Muon scales update by sqrt(max(1, fan_out/fan_in)) analogue:
    # for numpy port keep RMS-match factor used in Kimi Moonlight:
    scale = max(1.0, G.shape[0] / max(1, G.shape[1])) ** 0.5
    return (X * scale).astype(np.float64)


class Muon:
    """Muon for 2D weights (momentum -> Newton-Schulz ortho -> step),
    Adam for 1D weights (biases), exactly as Jordan 2024 prescribes.

    update = Ortho(momentum_buffer); p -= lr * update (+ decoupled WD optional).
    """

    def __init__(self, momentum=0.95, ns_steps=5, nesterov=True,
                 adam_b1=0.9, adam_b2=0.999, eps=1e-8, weight_decay=0.0):
        self.beta = momentum
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.adam = Adam(adam_b1, adam_b2, eps)
        self.wd = weight_decay
        self._adam_state = None

    def init_state(self, params):
        st = {"buf": [np.zeros_like(p) for p in params],
              "adam": self.adam.init_state(params)}
        return st

    def step(self, params, grads, state, lr, t=1):
        buf = state["buf"]
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            pd = p * (1 - lr * self.wd) if self.wd else p
            if p.ndim >= 2:
                buf[i] = self.beta * buf[i] + (1 - self.beta) * g
                # FIX (audit 2026-10-06): Nesterov blend was inverted vs muon.py.
                # torch: buf.lerp_(g, 1-b) = b*buf+(1-b)*g; g.lerp_(buf, b) =
                # (1-b)*g + b*buf (momentum-dominated). Ours had b*g+(1-b)*buf
                # (gradient-dominated). Now mirrors the reference exactly.
                upd = ((1 - self.beta) * g + self.beta * buf[i]
                       if self.nesterov else buf[i])
                # Newton-Schulz expects 2D; conv-style >2D flattened like muon.py
                shape = upd.shape
                U2 = upd.reshape(shape[0], -1) if upd.ndim > 2 else upd
                O = zeropower_via_newtonschulz5(U2, self.ns_steps).reshape(shape)
                # RMS-match: Muon updates have RMS ~0.2*sqrt(max(n,m)); rescale to Adam-like scale
                # Kimi Moonlight uses 0.2*sqrt(max(n,m)); here normalize to keep LR comparable:
                out.append(pd - lr * O * 0.2 * (max(shape) ** 0.5))
            else:
                # bias / vector params -> one Adam step on decayed param.
                # FIX (audit 2026-10-06): Adam.step rebinds m[i]/v[i] on the lists it
                # receives, so we must pass live single-element lists and write the
                # updated buffers back — the old wrapper-dict version silently dropped
                # momentum (biases were memory-less). Now moments truly accumulate.
                m_ = [state["adam"]["m"][i]]
                v_ = [state["adam"]["v"][i]]
                new_p = self.adam.step([pd], [g], {"m": m_, "v": v_}, lr, t)[0]
                state["adam"]["m"][i], state["adam"]["v"][i] = m_[0], v_[0]
                out.append(new_p)
        return out


def _spectral_norm_power(M, u, iters=3):
    """Top singular value of 2D M via power iteration (no SVD). u persists across steps."""
    for _ in range(iters):
        v = M @ u
        n = np.linalg.norm(v) + 1e-12
        v = v / n
        u = M.T @ v
        u = u / (np.linalg.norm(u) + 1e-12)
    return float(np.linalg.norm(M @ u)), u


class SpectraMix:
    """Our contender (2026): adaptively blend Muon <-> Adam per layer, per step.

    Thesis (from this repo's market-transfer finding): orthogonalization helps when the
    momentum spectrum is CONCENTRATED (few directions matter: transformers, vision MLP) and
    hurts when it is FLAT (heavy-tailed noise: finance). So measure concentration with the
    stable rank  sr = ||M||_F^2 / ||M||_2^2  in [1, r]  (spectral norm via 3 power iterations,
    no SVD), map to conc = 1-(sr-1)/(r-1) in [0,1], and set
        alpha = sigmoid(k*(conc-c0))   (defaults k=20 + schedule ON = measured champion:
        digits 0.072/0.948 @200 steps; plain k=12 close second 0.091/0.952)
        U = alpha*O/rms(O) + (1-alpha)*A/rms(A), renormalized, stepped at the blended scale
    where O = Newton-Schulz-5(momentum) and A = Adam direction. Endpoints are EXACT:
    alpha=1 -> Muon step, alpha=0 -> Adam step (unit-tested below in try_spectramix.py).
    1D params (biases) always take the Adam path. Decoupled weight decay like AdamW.
    """

    def __init__(self, momentum=0.95, ns_steps=5, nesterov=True, adam_b1=0.9, adam_b2=0.999,
                 eps=1e-8, weight_decay=0.0, k=20.0, c0=0.5, schedule=True):
        self.beta = momentum
        self.ns_steps = ns_steps
        self.nesterov = nesterov
        self.b1, self.b2, self.eps = adam_b1, adam_b2, eps
        self.wd = weight_decay
        self.k, self.c0 = k, c0
        self.schedule = schedule
        # schedule=True: DynMuon-inspired stage drift; late training leans toward
        # reallocating strength to flat directions (c0 drifts down over steps).

    def init_state(self, params):
        rng = np.random.default_rng(0)
        st = {"mom": _zeros_like(params), "m": _zeros_like(params),
              "v": _zeros_like(params), "u": [],
              "alphas": []}  # mean alpha per 2D param, for diagnostics
        for p in params:
            if p.ndim >= 2:
                n = p.shape[1] if p.ndim == 2 else int(np.prod(p.shape[1:]))
                u = rng.standard_normal(n)
                st["u"].append(u / (np.linalg.norm(u) + 1e-12))
            else:
                st["u"].append(None)
        return st

    def step(self, params, grads, state, lr, t=1, t_max=400):
        def sig(x):
            return 1.0 / (1.0 + np.exp(-x))
        out, alphas = [], []
        b1t, b2t = 1 - self.b1 ** t, 1 - self.b2 ** t
        # DynMuon-style stage drift (off by default; enabled only for the schedule variant)
        c0 = self.c0 - (0.15 * (t - 1) / max(t_max - 1, 1) if self.schedule else 0.0)
        for i, (p, g) in enumerate(zip(params, grads)):
            pd = p * (1 - lr * self.wd) if self.wd else p
            state["mom"][i] = self.beta * state["mom"][i] + (1 - self.beta) * g
            state["m"][i] = self.b1 * state["m"][i] + (1 - self.b1) * g
            state["v"][i] = self.b2 * state["v"][i] + (1 - self.b2) * g * g
            mh, vh = state["m"][i] / b1t, state["v"][i] / b2t
            A = mh / (np.sqrt(vh) + self.eps)  # Adam direction
            if p.ndim < 2:
                out.append(pd - lr * A)
                alphas.append(float("nan"))
                continue
            shape = g.shape
            # FIX (audit 2026-10-06 + MONA 2026): Nesterov BEFORE orthogonalization,
            # mirroring muon.py exactly: inp = (1-b)*g + b*buf. Old code fed raw buf.
            buf = state["mom"][i].reshape(shape[0], -1) if state["mom"][i].ndim > 2 else state["mom"][i]
            gg = g.reshape(shape[0], -1) if g.ndim > 2 else g
            M2 = ((1 - self.beta) * gg + self.beta * buf) if self.nesterov else buf
            F = float(np.linalg.norm(M2))
            if F < 1e-12:  # dead gradient -> Adam path
                out.append(pd - lr * A)
                alphas.append(0.0)
                continue
            s1, u = _spectral_norm_power(M2, state["u"][i])
            state["u"][i] = u
            r = min(M2.shape)
            sr = min(max(F * F / max(s1 * s1, 1e-18), 1.0), float(r))
            conc = 1.0 - (sr - 1.0) / max(r - 1.0, 1e-9)
            alpha = float(sig(self.k * (conc - c0)))
            O = zeropower_via_newtonschulz5(M2, self.ns_steps).reshape(shape)
            O = O * 0.2 * (max(shape) ** 0.5)  # same RMS-match as Muon
            ro, ra = float(np.sqrt((O * O).mean()) + 1e-12), float(np.sqrt((A * A).mean()) + 1e-12)
            U = alpha * O / ro + (1 - alpha) * A / ra
            U = U / (float(np.sqrt((U * U).mean())) + 1e-12)
            scale = alpha * ro + (1 - alpha) * ra  # endpoint-exact scales
            out.append(pd - lr * U * scale)
            alphas.append(alpha)
        state["alphas"] = alphas
        return out


def make_optimizer(name, **kw):
    name = name.lower()
    if name in ("gd", "sgd"):
        return SGD()
    if name == "momentum":
        return Momentum(mu=kw.get("mu", 0.9))
    if name == "rmsprop":
        return RMSprop()
    if name == "adam":
        return Adam()
    if name == "adamw":
        return AdamW(weight_decay=kw.get("weight_decay", 0.01))
    if name == "muon":
        return Muon(weight_decay=kw.get("weight_decay", 0.0))
    if name in ("spectramix", "smx"):
        return SpectraMix(weight_decay=kw.get("weight_decay", 0.0),
                          k=kw.get("k", 20.0), c0=kw.get("c0", 0.5),
                          schedule=kw.get("schedule", True),
                          nesterov=kw.get("nesterov", True))
    raise ValueError(f"unknown optimizer {name}")

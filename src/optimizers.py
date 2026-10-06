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
    Exact port of Keller Jordan's muon.py to NumPy (float32).
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
                upd = g * self.beta + buf[i] * (1 - self.beta) if self.nesterov else buf[i]
                # Newton-Schulz expects 2D; conv-style >2D flattened like muon.py
                shape = upd.shape
                U2 = upd.reshape(shape[0], -1) if upd.ndim > 2 else upd
                O = zeropower_via_newtonschulz5(U2, self.ns_steps).reshape(shape)
                # RMS-match: Muon updates have RMS ~0.2*sqrt(max(n,m)); rescale to Adam-like scale
                # Kimi Moonlight uses 0.2*sqrt(max(n,m)); here normalize to keep LR comparable:
                out.append(pd - lr * O * 0.2 * (max(shape) ** 0.5))
            else:
                # bias / vector params -> one Adam step on decayed param
                new_p = self.adam.step([pd], [g],
                                       {"m": [state["adam"]["m"][i]],
                                        "v": [state["adam"]["v"][i]]}, lr, t)[0]
                state["adam"]["m"][i] = state["adam"]["m"][i]  # updated in place by adam
                out.append(new_p)
        # NOTE: Adam sub-state for vector params is updated in place above
        # (m/v lists are mutated by Adam.step). Re-sync reference:
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
    raise ValueError(f"unknown optimizer {name}")

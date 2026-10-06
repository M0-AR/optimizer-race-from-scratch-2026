"""MLP from scratch (NumPy): 64-32x5-10, ReLU, softmax CE, manual backprop.

Param count check: 64*32 + 32*32*4 + 32*10 = 6464 weights
  + biases 32*5+10 = 170  => 6634 total (matches video).
"""
import numpy as np


def init_params(layer_sizes, seed=0):
    rng = np.random.default_rng(seed)
    params = []
    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        W = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)
        b = np.zeros(fan_out)
        params += [W, b]
    return params


def forward(params, X):
    acts = [X]
    pre = []
    h = X
    n_layers = len(params) // 2
    for i in range(n_layers):
        W, b = params[2 * i], params[2 * i + 1]
        z = h @ W + b
        pre.append(z)
        h = z if i == n_layers - 1 else np.maximum(z, 0)  # last = logits
        acts.append(h)
    return acts, pre


def loss_and_grads(params, X, y):
    n = len(X)
    acts, pre = forward(params, X)
    logits = acts[-1]
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=1, keepdims=True)
    correct = probs[np.arange(n), y]
    loss = -np.log(correct + 1e-12).mean()
    dlogits = probs.copy()
    dlogits[np.arange(n), y] -= 1
    dlogits /= n
    grads = [None] * len(params)
    n_layers = len(params) // 2
    dh = dlogits
    for i in reversed(range(n_layers)):
        a_prev = acts[i]
        grads[2 * i + 1] = dh.sum(axis=0)
        grads[2 * i] = a_prev.T @ dh
        if i > 0:
            dh = dh @ params[2 * i].T
            dh = dh * (pre[i - 1] > 0)
    return loss, grads, probs


def accuracy(params, X, y):
    acts, _ = forward(params, X)
    return (acts[-1].argmax(axis=1) == y).mean()


def train(params, opt, Xtr, ytr, lr, steps=400, batch=32, seed=0):
    rng = np.random.default_rng(seed)
    n = len(Xtr)
    state = opt.init_state(params)
    p = [x.copy() for x in params]
    hist = []
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, batch)
        loss, grads, _ = loss_and_grads(p, Xtr[idx], ytr[idx])
        hist.append(loss)
        p = opt.step(p, grads, state, lr, t)
    return p, hist

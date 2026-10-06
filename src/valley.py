"""Part 1: the long narrow valley. L = 0.5*x^2 + 50*y^2.

Reverse-engineered from the video: start (-10, 1) loss 100,
grad = (-10, +100), so curvatures a=1, b=100 (condition number 100).
Steep direction sets the LR speed limit: lr<0.02 stable, lr=0.021 explodes.
"""
import numpy as np


def loss(p):
    x, y = p[0], p[1]
    return 0.5 * x * x + 50.0 * y * y


def grad(p):
    return np.array([p[0], 100.0 * p[1]])


def run(opt, lr, steps=1000, start=(-10.0, 1.0)):
    from .optimizers import SGD  # local import safe
    params = [np.array(start, dtype=float)]
    state = opt.init_state(params)
    losses = []
    for t in range(1, steps + 1):
        g = [grad(params[0])]
        # optimizers expect list of arrays; our valley param is one 1-D vector
        # so wrap elementwise: use raw SGD-style manual step via optimizer on split?
        # Simplest: call optimizer on the single vector param directly.
        params = opt.step(params, g, state, lr, t)
        losses.append(loss(params[0]))
        if not np.isfinite(losses[-1]):
            break
    return params[0], losses


def count_working_lrs(opt_fn, lrs, steps=1000, target=1e-3):
    ok = 0
    for lr in lrs:
        _, losses = run(opt_fn(), lr, steps)
        if losses and losses[-1] < target:
            ok += 1
    return ok

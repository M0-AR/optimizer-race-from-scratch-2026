"""Data: sklearn digits (1797 x 8x8) -> 1297 train / 500 test, pixels/16 in [0,1].

Matches video exactly: 1297 train pictures + 500 hidden test pictures.
"""
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np


def load_digits_split(test_size=500, random_state=0):
    d = load_digits()
    X = d.data.astype(float) / 16.0
    y = d.target.astype(int)
    # deterministic: first 1297 train, last 500 test (stratified shuffle with fixed seed)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    return (Xtr, ytr), (Xte, yte)

#!/usr/bin/env python3
"""Choose explore or exploit action."""


import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Choose explore or exploit action."""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(4)
    else:
        return np.argmax(Q[state])

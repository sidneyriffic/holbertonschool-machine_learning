#!/usr/bin/env python3
"""Initialize Q-table"""


import numpy as np


def q_init(env):
    """Initialize Q-table"""
    return np.zeros((env.observation_space.n, env.action_space.n))

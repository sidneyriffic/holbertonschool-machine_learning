#!/usr/bin/env python3
"""Set up vanilla RNN cell"""


import numpy as np


class RNNCell:
    """A Basic RNN Cell"""
    def __init__(self, i, h, o):
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Do a forward prop time step"""
        concat = np.concatenate((h_prev, x_t), 1)
        next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        out = np.exp(np.matmul(next, self.Wy) + self.by)
        return next, out / out.sum(axis=1, keepdims=True)

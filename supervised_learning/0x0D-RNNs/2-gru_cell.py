#!/usr/bin/env python3
"""Initialization and forward time step for GRU Cell"""


import numpy as np


class GRUCell:
    """A GRU Cell"""
    def __init__(self, i, h, o):
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Do a forward prop time step"""
        concat = np.concatenate((h_prev, x_t), 1)
        ex = np.exp(np.matmul(concat, self.Wr) + self.br)
        ressig = ex / (1 + ex)
        ex = np.exp(np.matmul(concat, self.Wz) + self.bz)
        upsig = ex / (1 + ex)
        concat = np.concatenate((ressig * h_prev, x_t), 1)
        active = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        next = (1 - upsig) * h_prev + upsig * active
        out = np.exp(np.matmul(next, self.Wy) + self.by)
        return next, out / out.sum(axis=1, keepdims=True)

#!/usr/bin/env python3
"""Initialization and forward time step for LSTM Cell"""


import numpy as np


def sigmoid(var):
    """Do sigmoid on var"""
    ex = np.exp(var)
    return ex / (1 + ex)


class LSTMCell:
    """An LSTM Cell"""
    def __init__(self, i, h, o):
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Do a forward prop time step"""
        concat = np.concatenate((h_prev, x_t), 1)
        forgetsig = sigmoid(np.matmul(concat, self.Wf) + self.bf)
        middlesig = sigmoid(np.matmul(concat, self.Wu) + self.bu)
        active = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        outsig = sigmoid(np.matmul(concat, self.Wo) + self.bo)
        nextc = forgetsig * c_prev + middlesig * active
        nexth = outsig * np.tanh(nextc)
        out = np.exp(np.matmul(nexth, self.Wy) + self.by)
        return nexth, nextc, out / out.sum(axis=1, keepdims=True)

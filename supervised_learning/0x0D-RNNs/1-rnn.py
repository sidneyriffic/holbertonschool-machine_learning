#!/usr/bin/env python3
"""Perform forward propagation for an RNN"""


import numpy as np


def rnn(rnn_cell, X, h_0):
    """Perform forward propagation for an RNN"""
    h_hist = np.ndarray((X.shape[0] + 1, X.shape[1], h_0.shape[1]))
    y_hist = np.ndarray((X.shape[0], X.shape[1], rnn_cell.Wy.shape[1]))
    h_hist[0] = h_0
    for time in range(X.shape[0]):
        h = h_hist[time]
        h_hist[time + 1], y_hist[time] = rnn_cell.forward(h, X[time])
    return h_hist, y_hist

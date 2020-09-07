#!/usr/bin/env python3
"""Perform forward propagation for a bidirectional RNN"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Perform forward propagation for a bidirectional RNN"""
    h_size = h_0.shape[1]
    h_hist = np.ndarray((X.shape[0] + 1, X.shape[1], h_size * 2))
    y_hist = np.ndarray((X.shape[0], X.shape[1], bi_cell.Wy.shape[1]))
    h_hist[0, :, :h_size] = h_0
    h_hist[-1, :, h_size:] = h_t
    X_rev = np.flip(X, 0)
    for time in range(X.shape[0]):
        h = h_hist[time]
        h_hist[time + 1, :, :h_size] = bi_cell.forward(h[:, :h_size], X[time])
        hb = h_hist[-time]
        h_hist[-(time + 1), :, h_size:] = bi_cell.backward(hb[:, h_size:],
                                                           X_rev[time])
    return h_hist[1:], bi_cell.output(h_hist[1:])

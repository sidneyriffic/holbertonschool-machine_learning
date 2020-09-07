#!/usr/bin/env python3
"""Perform forward propagation for a deep RNN"""


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


def deep_rnn(rnn_cells, X, h_0):
    """Perform forward propagation for a deep RNN"""
    h_hist = np.ndarray((X.shape[0] + 1, len(rnn_cells),
                         X.shape[1], h_0.shape[2]))
    h_hist[0] = h_0
    h_hist[:, 0], _ = rnn(rnn_cells[0], X, h_0[0])
    for cell in range(1, len(rnn_cells)):
        h_hist[:, cell], ys = rnn(rnn_cells[cell], h_hist[1:, cell - 1],
                                  h_0[cell])
    return h_hist, ys

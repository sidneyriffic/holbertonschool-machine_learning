#!/usr/bin/env python3
"""Calculate positional encoding for a transformer"""


import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculate positional encoding for a transformer"""
    pos_enc = np.ndarray((max_seq_len, dm))
    for i in range(max_seq_len):
        for j in range(dm):
            if j % 2:
                pos_enc[i][j] = np.cos(i / np.power(10000, (j - 1) / dm))
            else:
                pos_enc[i][j] = np.sin(i / np.power(10000, j / dm))
    return pos_enc

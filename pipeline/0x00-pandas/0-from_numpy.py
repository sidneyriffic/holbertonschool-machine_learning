#!/usr/bin/env python3
"""Create a pandas dataframe from a numpy array"""


import pandas as pd


def from_numpy(array):
    """Create a pandas dataframe from a numpy array"""
    cols = [chr(i) for i in range(ord('A'), ord('A') + array.shape[1])]
    rows = [i for i in range(array.shape[0])]
    return pd.DataFrame(array, rows, cols)

#!/usr/bin/env python3
"""Load data from a file into a pandas dataframe"""

import pandas as pd


def from_file(filename, delimiter):
    """Load data from a file into a pandas dataframe"""
    return pd.read_csv(filename, delimiter, header=0)

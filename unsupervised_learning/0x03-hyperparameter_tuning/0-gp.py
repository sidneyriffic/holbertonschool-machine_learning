#!/usr/bin/env python3
"""Initialize Gaussian Process"""


import numpy as np


class GaussianProcess:
    """Hold state and data of a gaussian process"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Radial Basis Function Kernel"""
        return pow(self.sigma_f, 2) * np.exp(pow(X1 - X2.T, 2) /
                                             -2 / pow(self.l, 2))

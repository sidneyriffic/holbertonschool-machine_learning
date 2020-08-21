#!/usr/bin/env python3
"""Predict mean and standard deviation of points in Gaussian Process"""


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

    def predict(self, X_s):
        """Predict mean and standard deviation of points in Gaussian Process"""
        K_s = self.kernel(X_s, self.X)
        K_i = np.linalg.inv(self.K)
        mu = np.matmul(np.matmul(K_s, K_i), self.Y)[:, 0]
        K_s2 = self.kernel(X_s, X_s)
        sigma = K_s2 - np.matmul(np.matmul(K_s, K_i), K_s.T)
        return mu, np.diagonal(sigma)

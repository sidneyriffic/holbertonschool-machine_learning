#!/usr/bin/env python3
"""Initialize Bayesian Optimization"""


import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Initialize Bayesian Optimization"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        self.f = f
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples)[:, None]
        self.xsi = xsi
        self.minimize = minimize
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)

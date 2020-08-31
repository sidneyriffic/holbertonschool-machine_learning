#!/usr/bin/env python3
"""Initialize Bayesian Optimization"""


import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Initialize Bayesian Optimization"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        self.f = f
        self.X_s = np.linspace(bounds[0], bounds[1])[:, None]
        self.xsi = xsi
        self.minimize = minimize
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.Y_init = Y_init

    def acquisition(self):
        """Calculate next best sample location"""
        fs, _ = self.gp.predict(self.gp.X)
        next_fs, vars = self.gp.predict(self.X_s)
        opt = np.min(fs)
        improves = opt - next_fs - self.xsi
        if not self.minimize:
            improve = -improves
        Z = improves / vars
        eis = improves * norm.cdf(Z) + vars * norm.pdf(Z)
        return self.X_s[np.argmax(eis)], eis

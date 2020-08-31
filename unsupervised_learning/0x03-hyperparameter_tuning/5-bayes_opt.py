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
        Z = improves / vars
        eis = improves * norm.cdf(Z) + vars * norm.pdf(Z)
        return self.X_s[np.argmax(eis)], eis

    def optimize(self, iterations=1000):
        """Optimize for black box function"""
        prev = None
        finalx = None
        finaly = None
        while iterations:
            maxei, eis = self.acquisition()
            new_y = self.f(maxei)
            if maxei == prev:
                break
            self.gp.update(maxei, new_y)
            pycodehack = finaly is None or self.minimize and finaly > new_y
            if ((pycodehack or not self.minimize and finaly < new_y)):
                finaly = new_y
                finalx = maxei
            prev = maxei
            iterations -= 1
        return finalx, finaly

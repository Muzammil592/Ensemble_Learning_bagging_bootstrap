"""Bayesian bootstrap bagging (bonus).

Instead of drawing points WITH replacement (integer counts), every point gets a
continuous random weight drawn from Dirichlet(1, ..., 1). All points are used in every
member, but with different importance. It is a smooth version of the bootstrap.

Consequence: no point is ever fully "out of bag", so OOB estimates are not available.
"""
import numpy as np
from sklearn.base import clone

from .bagging import BaggingClassifier, BaggingRegressor


def _dirichlet_weights(n, rng):
    return rng.dirichlet(np.ones(n)) * n          # mean weight 1, like ordinary samples


class BayesianBaggingClassifier(BaggingClassifier):
    def __init__(self, base_estimator=None, n_estimators=50, voting="hard", random_state=None):
        super().__init__(base_estimator=base_estimator, n_estimators=n_estimators,
                         max_samples=1.0, bootstrap=True, oob_score=False,
                         voting=voting, random_state=random_state)

    def fit(self, X, y):
        X, y = self._validate(X, y)
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.classes_ = np.unique(y)
        base = self.base_estimator or self._default_estimator()
        self.estimators_ = []
        for _ in range(self.n_estimators):
            est = clone(base)
            if "random_state" in est.get_params():
                est.set_params(random_state=int(rng.integers(0, 2**31 - 1)))
            est.fit(X, y, sample_weight=_dirichlet_weights(n, rng))
            self.estimators_.append(est)
        return self


class BayesianBaggingRegressor(BaggingRegressor):
    def __init__(self, base_estimator=None, n_estimators=50, random_state=None):
        super().__init__(base_estimator=base_estimator, n_estimators=n_estimators,
                         max_samples=1.0, bootstrap=True, oob_score=False,
                         random_state=random_state)

    def fit(self, X, y):
        X, y = self._validate(X, y)
        y = y.astype(float)
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        base = self.base_estimator or self._default_estimator()
        self.estimators_ = []
        for _ in range(self.n_estimators):
            est = clone(base)
            if "random_state" in est.get_params():
                est.set_params(random_state=int(rng.integers(0, 2**31 - 1)))
            est.fit(X, y, sample_weight=_dirichlet_weights(n, rng))
            self.estimators_.append(est)
        return self

"""Bagging (Bootstrap Aggregating) implemented from scratch.

Only the base learner (a scikit-learn decision tree) is borrowed. Everything else
is written here: bootstrap resampling, vote/average aggregation, out-of-bag (OOB)
estimation, and staged predictions.

Why it works: each tree is trained on a different bootstrap sample, so the trees
make different errors. Averaging many high-variance, weakly correlated models
lowers variance while leaving bias roughly unchanged.
"""
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .bootstrap import bootstrap_indices, oob_indices


class _BaseBagging:
    def __init__(self, base_estimator=None, n_estimators=50, max_samples=1.0,
                 bootstrap=True, oob_score=False, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    # ----- helpers -----
    def _default_estimator(self):
        raise NotImplementedError

    def _draw(self, n, rng):
        size = max(1, int(round(self.max_samples * n)))
        if self.bootstrap:                      # bagging: with replacement
            return bootstrap_indices(n, rng, size=size)
        return rng.choice(n, size=size, replace=False)   # pasting: without replacement

    def _validate(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be >= 1.")
        if not 0 < self.max_samples <= 1:
            raise ValueError("max_samples must be in (0, 1].")
        return X, y

    def _check_fitted(self):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("Call fit() before predict().")


class BaggingClassifier(_BaseBagging):
    """Bagging for classification with hard (majority) or soft (probability) voting."""

    def __init__(self, base_estimator=None, n_estimators=50, max_samples=1.0,
                 bootstrap=True, oob_score=False, voting="hard", random_state=None):
        super().__init__(base_estimator, n_estimators, max_samples,
                         bootstrap, oob_score, random_state)
        if voting not in ("hard", "soft"):
            raise ValueError("voting must be 'hard' or 'soft'.")
        self.voting = voting

    def _default_estimator(self):
        return DecisionTreeClassifier()

    def _scores(self, est, X):
        """Per-class scores from one tree, aligned to self.classes_."""
        out = np.zeros((len(X), len(self.classes_)))
        cols = np.searchsorted(self.classes_, est.classes_)  # tree may miss a class
        if self.voting == "soft":
            out[:, cols] = est.predict_proba(X)
        else:
            pred = np.searchsorted(self.classes_, est.predict(X))
            out[np.arange(len(X)), pred] = 1.0
        return out

    def fit(self, X, y):
        """Fit n_estimators trees on bootstrap (or pasted) samples; compute OOB estimates if requested."""
        X, y = self._validate(X, y)
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.classes_ = np.unique(y)
        base = self.base_estimator or self._default_estimator()

        self.estimators_, self.oob_indices_ = [], []
        oob_scores = np.zeros((n, len(self.classes_)))
        oob_counts = np.zeros(n)
        self.oob_error_curve_ = []

        for _ in range(self.n_estimators):
            idx = self._draw(n, rng)
            est = clone(base)
            if "random_state" in est.get_params():
                est.set_params(random_state=int(rng.integers(0, 2**31 - 1)))
            est.fit(X[idx], y[idx])
            self.estimators_.append(est)

            oob = oob_indices(idx, n)
            self.oob_indices_.append(oob)
            if self.oob_score and len(oob):
                oob_scores[oob] += self._scores(est, X[oob])
                oob_counts[oob] += 1
                covered = oob_counts > 0
                if covered.any():
                    pred = self.classes_[oob_scores[covered].argmax(axis=1)]
                    self.oob_error_curve_.append(float((pred != y[covered]).mean()))

        if self.oob_score:
            covered = oob_counts > 0
            self.oob_decision_function_ = oob_scores
            self.oob_coverage_ = float(covered.mean())
            pred = self.classes_[oob_scores[covered].argmax(axis=1)]
            self.oob_score_ = float((pred == y[covered]).mean())
            self.oob_error_ = 1.0 - self.oob_score_
        return self

    def _aggregate(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return sum(self._scores(est, X) for est in self.estimators_)

    def predict(self, X):
        """Predict class labels by aggregating the trees' votes."""
        scores = self._aggregate(X)          # raises if not fitted
        return self.classes_[scores.argmax(axis=1)]

    def predict_proba(self, X):
        """Class probabilities: the normalised sum of per-tree scores."""
        total = self._aggregate(X)
        return total / total.sum(axis=1, keepdims=True)

    def score(self, X, y):
        """Accuracy on (X, y)."""
        return float((self.predict(X) == np.asarray(y)).mean())

    def staged_predict(self, X):
        """Yield predictions after 1, 2, ..., n_estimators trees (for error curves)."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        total = np.zeros((len(X), len(self.classes_)))
        for est in self.estimators_:
            total += self._scores(est, X)
            yield self.classes_[total.argmax(axis=1)]


class BaggingRegressor(_BaseBagging):
    """Bagging for regression: the prediction is the mean over all trees."""

    def _default_estimator(self):
        return DecisionTreeRegressor()

    def fit(self, X, y):
        """Fit n_estimators trees on bootstrap (or pasted) samples; compute OOB predictions if requested."""
        X, y = self._validate(X, y)
        y = y.astype(float)
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        base = self.base_estimator or self._default_estimator()

        self.estimators_, self.oob_indices_ = [], []
        oob_sum, oob_counts = np.zeros(n), np.zeros(n)
        self.oob_error_curve_ = []

        for _ in range(self.n_estimators):
            idx = self._draw(n, rng)
            est = clone(base)
            if "random_state" in est.get_params():
                est.set_params(random_state=int(rng.integers(0, 2**31 - 1)))
            est.fit(X[idx], y[idx])
            self.estimators_.append(est)

            oob = oob_indices(idx, n)
            self.oob_indices_.append(oob)
            if self.oob_score and len(oob):
                oob_sum[oob] += est.predict(X[oob])
                oob_counts[oob] += 1
                covered = oob_counts > 0
                if covered.any():
                    pred = oob_sum[covered] / oob_counts[covered]
                    self.oob_error_curve_.append(float(np.mean((pred - y[covered]) ** 2)))

        if self.oob_score:
            covered = oob_counts > 0
            pred = oob_sum[covered] / oob_counts[covered]
            self.oob_prediction_ = np.where(covered, oob_sum / np.maximum(oob_counts, 1), np.nan)
            self.oob_coverage_ = float(covered.mean())
            self.oob_mse_ = float(np.mean((pred - y[covered]) ** 2))
            ss_res = np.sum((y[covered] - pred) ** 2)
            ss_tot = np.sum((y[covered] - y[covered].mean()) ** 2)
            self.oob_score_ = float(1 - ss_res / ss_tot)   # R^2
        return self

    def predict(self, X):
        """Predict the mean of all trees' predictions."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        return np.mean([est.predict(X) for est in self.estimators_], axis=0)

    def score(self, X, y):
        """R^2 on (X, y)."""
        y = np.asarray(y, dtype=float)
        pred = self.predict(X)
        return float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))

    def staged_predict(self, X):
        """Yield the ensemble prediction after 1, 2, ..., n_estimators trees."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        total = np.zeros(len(X))
        for i, est in enumerate(self.estimators_, start=1):
            total += est.predict(X)
            yield total / i

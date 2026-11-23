"""Random forests built on top of our own bagging code.

A random forest = bagging + random feature subsetting at every split.
Bagging alone gives trees that are still strongly correlated (they keep picking the
same dominant features). Restricting each split to a random subset of features
decorrelates the trees, so averaging them removes more variance.

What is ours: bootstrap resampling, voting/averaging, OOB estimates (inherited from
src/bagging.py) and feature importances. What is borrowed: the base tree, whose
`max_features` argument performs the per-split random feature draw.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .bagging import BaggingClassifier, BaggingRegressor


class RandomForestClassifier(BaggingClassifier):
    """Random forest classifier.

    max_features : "sqrt" (default), "log2", an int, a float fraction, or None (all
                   features, which reduces to plain bagging).
    """

    def __init__(self, n_estimators=100, max_features="sqrt", max_depth=None,
                 min_samples_leaf=1, max_samples=1.0, oob_score=False,
                 voting="soft", random_state=None):
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        base = DecisionTreeClassifier(max_features=max_features, max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf)
        super().__init__(base_estimator=base, n_estimators=n_estimators,
                         max_samples=max_samples, bootstrap=True,
                         oob_score=oob_score, voting=voting,
                         random_state=random_state)

    @property
    def feature_importances_(self):
        """Mean impurity-based importance over all trees (sums to 1)."""
        self._check_fitted()
        imp = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        return imp / imp.sum()


class RandomForestRegressor(BaggingRegressor):
    """Random forest regressor (default max_features = 1/3 of the features)."""

    def __init__(self, n_estimators=100, max_features=1 / 3, max_depth=None,
                 min_samples_leaf=1, max_samples=1.0, oob_score=False,
                 random_state=None):
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        base = DecisionTreeRegressor(max_features=max_features, max_depth=max_depth,
                                     min_samples_leaf=min_samples_leaf)
        super().__init__(base_estimator=base, n_estimators=n_estimators,
                         max_samples=max_samples, bootstrap=True,
                         oob_score=oob_score, random_state=random_state)

    @property
    def feature_importances_(self):
        """Mean impurity-based importance over all trees (sums to 1)."""
        self._check_fitted()
        imp = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        return imp / imp.sum()

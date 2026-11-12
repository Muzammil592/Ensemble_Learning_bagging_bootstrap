"""Show that a single decision tree has high variance, and that this is exactly
what bagging (Day 2) will fix.

We train many trees, each on a different bootstrap resample of the SAME training
set, and look at how much their predictions disagree.
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .bootstrap import bootstrap_sample


def tree_variance_experiment(X_train, y_train, X_test, y_test,
                             n_trees: int = 100, seed: int = 0, max_depth=None):
    """Train n_trees trees on bootstrap resamples.

    Returns
    -------
    dict with:
      test_accs : (n_trees,) test accuracy of each tree
      preds     : (n_trees, n_test) predictions of each tree
      full_acc  : test accuracy of one tree trained on the full train set
    """
    rng = np.random.default_rng(seed)

    full_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    full_tree.fit(X_train, y_train)
    full_acc = float(full_tree.score(X_test, y_test))

    test_accs, preds = [], []
    for i in range(n_trees):
        Xb, yb, _ = bootstrap_sample(X_train, y_train, rng=rng)
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=i)
        tree.fit(Xb, yb)
        p = tree.predict(X_test)
        preds.append(p)
        test_accs.append(float((p == y_test).mean()))

    return {
        "test_accs": np.array(test_accs),
        "preds": np.array(preds),
        "full_acc": full_acc,
    }


def prediction_variance_per_point(preds: np.ndarray) -> np.ndarray:
    """Disagreement per test point across trees.

    For binary 0/1 predictions this is p(1-p) where p = share of trees voting 1:
    0 means all trees agree, 0.25 means a perfect 50/50 split.
    """
    p = preds.mean(axis=0)
    return p * (1 - p)

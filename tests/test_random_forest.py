"""Unit tests for random forests and the bias-variance decomposition. Run with:  pytest -q   (from the project root)."""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SkRFC
from sklearn.ensemble import RandomForestRegressor as SkRFR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (BaggingRegressor, RandomForestClassifier, RandomForestRegressor,  # noqa: E402
                 bias_variance_decomposition, friedman_f, make_friedman_data)
from utils import load_dataset, train_test_split_np  # noqa: E402


def _cls():
    X, y, _ = load_dataset("breast_cancer")
    return train_test_split_np(X, y, test_size=0.3, seed=0)


def _reg():
    X, y, _ = load_dataset("diabetes")
    return train_test_split_np(X, y, test_size=0.3, seed=0, stratify=False)


# ---------------- random forest classifier ----------------
def test_rf_shapes_and_proba():
    Xtr, Xte, ytr, yte = _cls()
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(Xtr, ytr)
    assert rf.predict(Xte).shape == yte.shape
    p = rf.predict_proba(Xte)
    assert p.shape == (len(Xte), 2) and np.allclose(p.sum(axis=1), 1)


def test_rf_passes_max_features_to_trees():
    Xtr, _, ytr, _ = _cls()
    rf = RandomForestClassifier(n_estimators=5, max_features=4, random_state=0).fit(Xtr, ytr)
    assert all(t.max_features == 4 for t in rf.estimators_)
    assert all(t.max_features_ == 4 for t in rf.estimators_)


def test_rf_is_deterministic():
    Xtr, Xte, ytr, _ = _cls()
    a = RandomForestClassifier(n_estimators=15, random_state=3).fit(Xtr, ytr).predict(Xte)
    b = RandomForestClassifier(n_estimators=15, random_state=3).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)


def test_rf_matches_sklearn_within_tolerance():
    Xtr, Xte, ytr, yte = _cls()
    ours = RandomForestClassifier(n_estimators=100, random_state=0).fit(Xtr, ytr).score(Xte, yte)
    theirs = SkRFC(n_estimators=100, random_state=0).fit(Xtr, ytr).score(Xte, yte)
    assert abs(ours - theirs) < 0.03


def test_rf_oob_close_to_test():
    Xtr, Xte, ytr, yte = _cls()
    rf = RandomForestClassifier(n_estimators=150, oob_score=True, random_state=0).fit(Xtr, ytr)
    assert abs(rf.oob_score_ - rf.score(Xte, yte)) < 0.05


def test_feature_importances_sum_to_one():
    Xtr, _, ytr, _ = _cls()
    rf = RandomForestClassifier(n_estimators=30, random_state=0).fit(Xtr, ytr)
    imp = rf.feature_importances_
    assert imp.shape == (Xtr.shape[1],)
    assert abs(imp.sum() - 1) < 1e-9 and (imp >= 0).all()


def test_rf_trees_are_more_diverse_than_bagging_trees():
    """Feature subsetting should make individual trees disagree more."""
    from src import BaggingClassifier
    Xtr, Xte, ytr, _ = _cls()
    def disagreement(model):
        P = np.array([t.predict(Xte) for t in model.estimators_])
        return np.mean(P != np.round(P.mean(axis=0)))
    bag = BaggingClassifier(n_estimators=40, voting="soft", random_state=0).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=40, max_features=3, random_state=0).fit(Xtr, ytr)
    assert disagreement(rf) >= disagreement(bag)


def test_rf_regressor_default_and_score():
    Xtr, Xte, ytr, yte = _reg()
    rf = RandomForestRegressor(n_estimators=60, random_state=0).fit(Xtr, ytr)
    assert rf.predict(Xte).shape == yte.shape
    assert rf.score(Xte, yte) > 0.3


def test_rf_regressor_matches_sklearn_within_tolerance():
    Xtr, Xte, ytr, yte = _reg()
    ours = RandomForestRegressor(n_estimators=100, random_state=0).fit(Xtr, ytr).score(Xte, yte)
    theirs = SkRFR(n_estimators=100, max_features=1 / 3, random_state=0).fit(Xtr, ytr).score(Xte, yte)
    assert abs(ours - theirs) < 0.08


# ---------------- bias-variance ----------------
def test_friedman_data_shapes():
    X, y = make_friedman_data(50, rng=0, n_features=10, noise=1.0)
    assert X.shape == (50, 10) and y.shape == (50,)
    assert friedman_f(X).shape == (50,)


def test_decomposition_identity_holds():
    """mean squared error vs the truth must equal bias^2 + variance exactly."""
    res = bias_variance_decomposition(lambda s: BaggingRegressor(n_estimators=5, random_state=s),
                                      n_rounds=15, n_train=80, n_test=100, seed=1)
    assert abs(res["mse_vs_truth"] - (res["bias2"] + res["variance"])) < 1e-9
    assert abs(res["expected_error"] - (res["bias2"] + res["variance"] + res["noise"])) < 1e-9


def test_constant_model_has_zero_variance():
    class Const:
        def fit(self, X, y):
            return self
        def predict(self, X):
            return np.zeros(len(X))
    res = bias_variance_decomposition(lambda s: Const(), n_rounds=10, n_train=30,
                                      n_test=50, seed=0)
    assert res["variance"] == 0.0 and res["bias2"] > 0


def test_ensembling_reduces_variance():
    from sklearn.tree import DecisionTreeRegressor
    single = bias_variance_decomposition(lambda s: DecisionTreeRegressor(random_state=s),
                                         n_rounds=30, n_train=100, n_test=150, seed=2)
    forest = bias_variance_decomposition(
        lambda s: RandomForestRegressor(n_estimators=15, max_features=0.5, random_state=s),
        n_rounds=30, n_train=100, n_test=150, seed=2)
    assert forest["variance"] < single["variance"]

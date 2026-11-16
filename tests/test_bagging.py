"""Unit tests. Run with:  pytest -q   (from the project root)."""
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import BaggingClassifier as SkBaggingClassifier
from sklearn.ensemble import BaggingRegressor as SkBaggingRegressor
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (BaggingClassifier, BaggingRegressor, bootstrap_indices,  # noqa: E402
                 oob_indices)
from utils import load_dataset, train_test_split_np  # noqa: E402


def _cls_data():
    X, y, _ = load_dataset("breast_cancer")
    return train_test_split_np(X, y, test_size=0.3, seed=0)


def _reg_data():
    X, y, _ = load_dataset("diabetes")
    return train_test_split_np(X, y, test_size=0.3, seed=0, stratify=False)


# ---------------- bootstrap helpers ----------------
def test_bootstrap_indices_range_and_size():
    idx = bootstrap_indices(100, rng=1)
    assert idx.shape == (100,)
    assert idx.min() >= 0 and idx.max() < 100


def test_oob_is_complement_of_drawn():
    idx = bootstrap_indices(200, rng=2)
    oob = oob_indices(idx, 200)
    assert set(oob).isdisjoint(set(idx))
    assert len(set(oob) | set(idx)) == 200


# ---------------- classifier ----------------
def test_classifier_shapes_and_labels():
    Xtr, Xte, ytr, yte = _cls_data()
    clf = BaggingClassifier(n_estimators=10, random_state=0).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    assert pred.shape == yte.shape
    assert set(pred) <= set(np.unique(ytr))
    proba = clf.predict_proba(Xte)
    assert proba.shape == (len(Xte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_classifier_is_deterministic_with_seed():
    Xtr, Xte, ytr, _ = _cls_data()
    a = BaggingClassifier(n_estimators=15, random_state=7).fit(Xtr, ytr).predict(Xte)
    b = BaggingClassifier(n_estimators=15, random_state=7).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)


def test_classifier_beats_or_matches_single_tree():
    Xtr, Xte, ytr, yte = _cls_data()
    tree = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).score(Xte, yte)
    bag = BaggingClassifier(n_estimators=50, random_state=0).fit(Xtr, ytr).score(Xte, yte)
    assert bag >= tree - 0.01


def test_classifier_matches_sklearn_within_tolerance():
    Xtr, Xte, ytr, yte = _cls_data()
    ours = BaggingClassifier(n_estimators=50, random_state=0).fit(Xtr, ytr).score(Xte, yte)
    theirs = SkBaggingClassifier(DecisionTreeClassifier(), n_estimators=50,
                                 random_state=0).fit(Xtr, ytr).score(Xte, yte)
    assert abs(ours - theirs) < 0.03


def test_soft_voting_works():
    Xtr, Xte, ytr, yte = _cls_data()
    clf = BaggingClassifier(n_estimators=30, voting="soft", random_state=0).fit(Xtr, ytr)
    assert clf.score(Xte, yte) > 0.9


def test_classifier_oob_close_to_test_accuracy():
    Xtr, Xte, ytr, yte = _cls_data()
    clf = BaggingClassifier(n_estimators=100, oob_score=True, random_state=0).fit(Xtr, ytr)
    assert 0 <= clf.oob_score_ <= 1
    assert abs(clf.oob_score_ - clf.score(Xte, yte)) < 0.05
    assert clf.oob_coverage_ > 0.99          # with 100 trees nearly every point is OOB once
    assert len(clf.oob_error_curve_) > 0


def test_staged_predict_last_equals_predict():
    Xtr, Xte, ytr, _ = _cls_data()
    clf = BaggingClassifier(n_estimators=12, random_state=0).fit(Xtr, ytr)
    stages = list(clf.staged_predict(Xte))
    assert len(stages) == 12
    assert np.array_equal(stages[-1], clf.predict(Xte))


def test_pasting_uses_no_duplicates():
    Xtr, _, ytr, _ = _cls_data()
    clf = BaggingClassifier(n_estimators=5, bootstrap=False, max_samples=0.6,
                            oob_score=True, random_state=0).fit(Xtr, ytr)
    n = len(Xtr)
    for oob in clf.oob_indices_:
        assert abs(len(oob) - (n - round(0.6 * n))) <= 1   # exactly the unused points


def test_predict_before_fit_raises():
    try:
        BaggingClassifier().predict(np.zeros((2, 3)))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_bad_arguments_raise():
    for kwargs in ({"voting": "bad"},):
        try:
            BaggingClassifier(**kwargs)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")
    Xtr, _, ytr, _ = _cls_data()
    try:
        BaggingClassifier(n_estimators=0).fit(Xtr, ytr)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------------- regressor ----------------
def test_regressor_shapes_and_r2():
    Xtr, Xte, ytr, yte = _reg_data()
    reg = BaggingRegressor(n_estimators=30, random_state=0).fit(Xtr, ytr)
    assert reg.predict(Xte).shape == yte.shape
    assert reg.score(Xte, yte) > 0.2


def test_regressor_matches_sklearn_within_tolerance():
    Xtr, Xte, ytr, yte = _reg_data()
    ours = BaggingRegressor(n_estimators=50, random_state=0).fit(Xtr, ytr).score(Xte, yte)
    theirs = SkBaggingRegressor(n_estimators=50, random_state=0).fit(Xtr, ytr).score(Xte, yte)
    assert abs(ours - theirs) < 0.08


def test_regressor_oob_r2_is_reasonable():
    Xtr, Xte, ytr, yte = _reg_data()
    reg = BaggingRegressor(n_estimators=100, oob_score=True, random_state=0).fit(Xtr, ytr)
    assert abs(reg.oob_score_ - reg.score(Xte, yte)) < 0.15
    assert reg.oob_mse_ > 0


def test_regressor_staged_last_equals_predict():
    Xtr, Xte, ytr, _ = _reg_data()
    reg = BaggingRegressor(n_estimators=8, random_state=0).fit(Xtr, ytr)
    stages = list(reg.staged_predict(Xte))
    assert np.allclose(stages[-1], reg.predict(Xte))

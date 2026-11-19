"""Day 4 unit tests. Run with:  pytest -q   (from the project root)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import (BayesianBaggingClassifier, BayesianBaggingRegressor,  # noqa: E402
                 RandomForestRegressor, accuracy, bootstrap_ci, bootstrap_mean_ci,
                 coverage_experiment, cv_predict, interval_coverage, mae, make_friedman_data,
                 mse, oob_residual_interval, paired_bootstrap_diff_ci, r2, rmse,
                 tree_spread_interval)
from utils import kfold_indices, load_dataset, train_test_split_np  # noqa: E402


# ---------------- metrics ----------------
def test_metrics_known_values():
    y, p = np.array([1, 2, 3, 4]), np.array([1, 2, 3, 6])
    assert accuracy(y, p) == 0.75
    assert abs(mse(y, p) - 1.0) < 1e-12
    assert abs(rmse(y, p) - 1.0) < 1e-12
    assert abs(mae(y, p) - 0.5) < 1e-12
    assert r2(y, y) == 1.0


# ---------------- bootstrap CIs ----------------
def test_mean_ci_contains_estimate_and_is_ordered():
    v = np.random.default_rng(0).normal(size=200)
    ci = bootstrap_mean_ci(v, n_boot=500, rng=1)
    assert ci["lo"] < ci["estimate"] < ci["hi"]
    assert len(ci["boot"]) == 500


def test_ci_is_deterministic_with_seed():
    v = np.random.default_rng(0).integers(0, 2, 100)
    a = bootstrap_mean_ci(v, rng=5)
    b = bootstrap_mean_ci(v, rng=5)
    assert a["lo"] == b["lo"] and a["hi"] == b["hi"]


def test_ci_narrows_with_more_data():
    rng = np.random.default_rng(1)
    small = bootstrap_mean_ci(rng.binomial(1, 0.8, 50), rng=2)
    large = bootstrap_mean_ci(rng.binomial(1, 0.8, 800), rng=2)
    assert (large["hi"] - large["lo"]) < (small["hi"] - small["lo"])


def test_perfect_predictions_give_degenerate_ci():
    ci = bootstrap_mean_ci(np.ones(80), rng=0)
    assert ci["lo"] == ci["hi"] == 1.0


def test_generic_ci_agrees_with_vectorised_for_accuracy():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 150)
    p = np.where(rng.random(150) < 0.85, y, 1 - y)
    fast = bootstrap_mean_ci((y == p).astype(float), n_boot=1500, rng=1)
    slow = bootstrap_ci(y, p, accuracy, n_boot=1500, rng=2)
    assert abs(fast["lo"] - slow["lo"]) < 0.03 and abs(fast["hi"] - slow["hi"]) < 0.03


def test_rmse_ci_contains_estimate():
    rng = np.random.default_rng(4)
    y = rng.normal(size=120)
    p = y + rng.normal(scale=0.5, size=120)
    ci = bootstrap_ci(y, p, rmse, n_boot=500, rng=0)
    assert ci["lo"] < ci["estimate"] < ci["hi"]


def test_coverage_is_close_to_nominal():
    pool = np.random.default_rng(0).binomial(1, 0.8, 5000)
    truth, cov = coverage_experiment(pool, [100], n_trials=300, n_boot=400, seed=0)
    assert abs(truth - pool.mean()) < 1e-12
    assert 0.90 <= cov[100] <= 0.99


def test_paired_diff_zero_for_identical_models():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 100)
    p = np.where(rng.random(100) < 0.8, y, 1 - y)
    d = paired_bootstrap_diff_ci(y, p, p, accuracy, n_boot=200, rng=0)
    assert d["estimate"] == 0 and d["lo"] == 0 and d["hi"] == 0
    assert d["excludes_zero"] is False


def test_paired_diff_detects_clearly_better_model():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 400)
    good = np.where(rng.random(400) < 0.95, y, 1 - y)
    bad = np.where(rng.random(400) < 0.60, y, 1 - y)
    d = paired_bootstrap_diff_ci(y, good, bad, accuracy, n_boot=500, rng=0)
    assert d["lo"] > 0 and d["excludes_zero"] is True


# ---------------- prediction intervals ----------------
def _friedman_fit():
    Xtr, ytr = make_friedman_data(400, rng=10, noise=1.0)
    Xte, yte = make_friedman_data(400, rng=11, noise=1.0)
    rf = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=0).fit(Xtr, ytr)
    return rf, Xte, yte, ytr


def test_oob_interval_has_reasonable_coverage():
    rf, Xte, yte, ytr = _friedman_fit()
    lo, hi = oob_residual_interval(rf, Xte, ytr, alpha=0.05)
    assert np.all(lo <= hi)
    cov, width = interval_coverage(yte, lo, hi)
    assert 0.88 <= cov <= 0.99 and width > 0


def test_oob_interval_is_tighter_than_tree_spread_at_similar_coverage():
    """Tree spread is conservative at low noise; the OOB interval is as good and narrower."""
    rf, Xte, yte, ytr = _friedman_fit()
    spread_cov, spread_w = interval_coverage(yte, *tree_spread_interval(rf, Xte))
    oob_cov, oob_w = interval_coverage(yte, *oob_residual_interval(rf, Xte, ytr))
    assert spread_cov >= 0.9 and oob_cov >= 0.88
    assert oob_w < 0.9 * spread_w


def test_oob_interval_requires_oob_score():
    Xtr, ytr = make_friedman_data(100, rng=0)
    rf = RandomForestRegressor(n_estimators=5, random_state=0).fit(Xtr, ytr)
    try:
        oob_residual_interval(rf, Xtr, ytr)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_interval_coverage_known_values():
    cov, width = interval_coverage(np.array([1, 2, 3, 4]), np.array([0, 0, 0, 0]),
                                   np.array([2, 2, 2, 2]))
    assert cov == 0.5 and width == 2.0


# ---------------- k-fold / cv_predict ----------------
def test_kfold_partitions_data():
    y = np.array([0] * 50 + [1] * 30)
    folds = kfold_indices(len(y), k=5, seed=0, y=y)
    all_test = np.concatenate([te for _, te in folds])
    assert sorted(all_test) == list(range(len(y)))
    for tr, te in folds:
        assert set(tr).isdisjoint(set(te))
        assert abs(y[te].mean() - 30 / 80) < 0.06        # stratified


def test_cv_predict_returns_one_prediction_per_sample():
    from sklearn.tree import DecisionTreeClassifier
    X, y, _ = load_dataset("wine")
    pred = cv_predict(lambda s: DecisionTreeClassifier(random_state=s), X, y, k=5, seed=0)
    assert pred.shape == y.shape and accuracy(y, pred) > 0.8


# ---------------- Bayesian bootstrap bagging ----------------
def test_bayesian_classifier_works_and_is_deterministic():
    X, y, _ = load_dataset("breast_cancer")
    Xtr, Xte, ytr, yte = train_test_split_np(X, y, test_size=0.3, seed=0)
    a = BayesianBaggingClassifier(n_estimators=20, voting="soft", random_state=1).fit(Xtr, ytr)
    b = BayesianBaggingClassifier(n_estimators=20, voting="soft", random_state=1).fit(Xtr, ytr)
    assert a.score(Xte, yte) > 0.9
    assert np.array_equal(a.predict(Xte), b.predict(Xte))
    assert a.predict_proba(Xte).shape == (len(Xte), 2)


def test_bayesian_members_differ():
    X, y, _ = load_dataset("breast_cancer")
    m = BayesianBaggingClassifier(n_estimators=10, random_state=0).fit(X, y)
    depths = {t.get_n_leaves() for t in m.estimators_}
    imps = {tuple(np.round(t.feature_importances_, 4)) for t in m.estimators_}
    assert len(imps) > 1 or len(depths) > 1


def test_bayesian_regressor_runs():
    X, y, _ = load_dataset("diabetes")
    Xtr, Xte, ytr, yte = train_test_split_np(X, y, test_size=0.3, seed=0, stratify=False)
    reg = BayesianBaggingRegressor(n_estimators=30, random_state=0).fit(Xtr, ytr)
    assert reg.predict(Xte).shape == yte.shape and reg.score(Xte, yte) > 0.2

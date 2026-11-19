"""Using the bootstrap to quantify uncertainty.

Two different questions, two different tools:

1. "How uncertain is my reported metric?"  -> percentile bootstrap confidence
   intervals (resample the TEST examples, recompute the metric).
2. "How uncertain is one prediction?"       -> prediction intervals. For ensembles we
   compare the spread across trees (captures model variance only) with an interval
   built from out-of-bag residuals (also captures noise in y).
"""
import numpy as np


# ------------------------------------------------------------------
# Confidence intervals for metrics
# ------------------------------------------------------------------
def bootstrap_mean_ci(values, n_boot=2000, alpha=0.05, rng=None):
    """Percentile bootstrap CI for the mean of per-example values (vectorised).

    Accuracy is the mean of a 0/1 "correct" vector, so this covers accuracy and, with
    squared errors, MSE.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(rng)
    boots = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(values.mean()), "lo": float(lo), "hi": float(hi),
            "se": float(boots.std(ddof=1)), "boot": boots}


def bootstrap_ci(y_true, y_pred, metric, n_boot=2000, alpha=0.05, rng=None):
    """Percentile bootstrap CI for any metric(y_true, y_pred) (RMSE, R^2, ...)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(rng)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(metric(y_true, y_pred)), "lo": float(lo), "hi": float(hi),
            "se": float(boots.std(ddof=1)), "boot": boots}


def paired_bootstrap_diff_ci(y_true, pred_a, pred_b, metric, n_boot=2000,
                             alpha=0.05, rng=None):
    """CI for metric(A) - metric(B) using the SAME resampled examples for both models.

    Pairing removes the example-to-example noise the two models share, so it is far
    more sensitive than comparing two separate intervals. If the CI excludes 0 the
    difference is unlikely to be resampling noise.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    n = len(y_true)
    rng = np.random.default_rng(rng)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = metric(y_true[idx], pred_a[idx]) - metric(y_true[idx], pred_b[idx])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"estimate": float(metric(y_true, pred_a) - metric(y_true, pred_b)),
            "lo": float(lo), "hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}


def coverage_experiment(correct_pool, test_sizes, n_trials=1000, n_boot=1000,
                        alpha=0.05, seed=0):
    """Check that the bootstrap CI for accuracy really contains the truth ~95% of the time.

    The "population" is a pool of per-example correct/incorrect outcomes; its mean is the
    true accuracy. For each test size n we draw n_trials iid test sets from the pool,
    build a percentile CI from each, and count how often it contains the true accuracy.
    """
    correct_pool = np.asarray(correct_pool, dtype=float)
    truth = correct_pool.mean()
    rng = np.random.default_rng(seed)
    out = {}
    for n in test_sizes:
        hits = 0
        for _ in range(n_trials):
            sample = correct_pool[rng.integers(0, len(correct_pool), size=n)]
            ci = bootstrap_mean_ci(sample, n_boot=n_boot, alpha=alpha, rng=rng)
            hits += ci["lo"] <= truth <= ci["hi"]
        out[int(n)] = hits / n_trials
    return truth, out


# ------------------------------------------------------------------
# Prediction intervals for ensemble regressors
# ------------------------------------------------------------------
def tree_spread_interval(model, X, alpha=0.05):
    """Percentiles of the individual trees' predictions.

    This only reflects disagreement between trees (model variance). It ignores the noise
    in y itself, so it is NOT a prediction interval and will under-cover.
    """
    X = np.asarray(X, dtype=float)
    preds = np.array([est.predict(X) for est in model.estimators_])
    return tuple(np.quantile(preds, [alpha / 2, 1 - alpha / 2], axis=0))


def oob_residual_interval(model, X, y_train, alpha=0.05):
    """Prediction interval = forest prediction + quantiles of the OOB residuals.

    Each training point's OOB prediction comes from trees that never saw it, so
    (y - OOB prediction) behaves like a genuine out-of-sample error. Their empirical
    quantiles give an interval that includes irreducible noise.
    Requires the model to be fitted with oob_score=True.
    """
    if not hasattr(model, "oob_prediction_"):
        raise ValueError("Model must be fitted with oob_score=True.")
    y_train = np.asarray(y_train, dtype=float)
    resid = y_train - model.oob_prediction_
    resid = resid[~np.isnan(resid)]
    q_lo, q_hi = np.quantile(resid, [alpha / 2, 1 - alpha / 2])
    pred = model.predict(X)
    return pred + q_lo, pred + q_hi


def interval_coverage(y_true, lo, hi):
    """Return (empirical coverage, mean interval width)."""
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_true >= lo) & (y_true <= hi))), float(np.mean(hi - lo))

"""Monte-Carlo bias-variance decomposition for regression models.

We use a synthetic problem (Friedman #1) where the true function f(x) is known, so
the decomposition is exact rather than estimated from noisy labels:

    E[(y - f_hat(x))^2] = noise^2 + bias^2(x) + variance(x)

    bias^2(x)   = (E_D[f_hat(x)] - f(x))^2        error from the model being wrong
    variance(x) = E_D[(f_hat(x) - E_D[f_hat(x)])^2]  error from sensitivity to the data

The expectation E_D is approximated by training the model on many independent
training sets D drawn from the same distribution.
"""
import numpy as np


def friedman_f(X):
    """True Friedman #1 function (only the first 5 features matter)."""
    return (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
            + 10 * X[:, 3] + 5 * X[:, 4])


def make_friedman_data(n, rng, n_features=10, noise=1.0):
    """Draw n samples: X ~ U(0,1)^p, y = f(X) + N(0, noise^2)."""
    rng = np.random.default_rng(rng)
    X = rng.uniform(size=(n, n_features))
    y = friedman_f(X) + rng.normal(0, noise, size=n)
    return X, y


def bias_variance_decomposition(model_factory, n_rounds=100, n_train=200,
                                n_test=500, n_features=10, noise=1.0, seed=0):
    """Estimate bias^2, variance and noise for a model class.

    Parameters
    ----------
    model_factory : callable(seed) -> unfitted model with fit(X, y) / predict(X).
    n_rounds      : number of independent training sets.

    Returns
    -------
    dict: bias2, variance, noise, mse_vs_truth (= bias2 + variance),
          expected_error (= bias2 + variance + noise).
    """
    rng = np.random.default_rng(seed)
    X_test = rng.uniform(size=(n_test, n_features))
    f_test = friedman_f(X_test)

    preds = np.empty((n_rounds, n_test))
    for r in range(n_rounds):
        X_tr, y_tr = make_friedman_data(n_train, rng, n_features, noise)
        model = model_factory(int(rng.integers(0, 2**31 - 1)))
        model.fit(X_tr, y_tr)
        preds[r] = model.predict(X_test)

    mean_pred = preds.mean(axis=0)
    bias2 = float(np.mean((mean_pred - f_test) ** 2))
    variance = float(np.mean(preds.var(axis=0)))
    mse_vs_truth = float(np.mean((preds - f_test) ** 2))
    return {
        "bias2": bias2,
        "variance": variance,
        "noise": float(noise ** 2),
        "mse_vs_truth": mse_vs_truth,
        "expected_error": bias2 + variance + noise ** 2,
    }

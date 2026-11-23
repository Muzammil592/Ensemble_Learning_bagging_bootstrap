"""Core bootstrap utilities (NumPy only).

A bootstrap sample of size n is drawn *with replacement* from n original points.
Each point has probability (1 - 1/n)^n -> 1/e ≈ 0.368 of never being drawn, so on
average ~63.2% of the original points appear in a bootstrap sample. The ~36.8% left
out are the "out-of-bag" (OOB) points, which bagging uses as a free validation set.
"""
import numpy as np


def bootstrap_indices(n_samples: int, rng=None, size: int = None) -> np.ndarray:
    """Draw bootstrap indices (with replacement).

    Parameters
    ----------
    n_samples : number of points in the original dataset.
    rng : np.random.Generator, int seed, or None.
    size : size of the bootstrap sample (defaults to n_samples).
    """
    rng = np.random.default_rng(rng)
    size = n_samples if size is None else size
    return rng.integers(0, n_samples, size=size)


def bootstrap_sample(X, y, rng=None, size: int = None):
    """Return a bootstrap sample (X_b, y_b) plus the indices that were drawn."""
    X, y = np.asarray(X), np.asarray(y)
    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")
    idx = bootstrap_indices(len(X), rng=rng, size=size)
    return X[idx], y[idx], idx


def oob_indices(idx: np.ndarray, n_samples: int) -> np.ndarray:
    """Indices of original points that were NOT drawn in the bootstrap sample."""
    mask = np.ones(n_samples, dtype=bool)
    mask[idx] = False
    return np.where(mask)[0]


def unique_fraction(idx: np.ndarray, n_samples: int) -> float:
    """Fraction of original points that appear at least once in the sample."""
    return len(np.unique(idx)) / n_samples


def simulate_unique_fraction(n_samples: int, n_trials: int = 200, rng=None) -> float:
    """Mean unique fraction over many bootstrap draws."""
    rng = np.random.default_rng(rng)
    fracs = [unique_fraction(bootstrap_indices(n_samples, rng), n_samples)
             for _ in range(n_trials)]
    return float(np.mean(fracs))

"""Helpers for loading CSV datasets from the data/ folder."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_dataset(name: str = "breast_cancer"):
    """Load a dataset from data/<name>.csv.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    feature_names : list[str]
    """
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    features = df.drop(columns=["target"])
    return features.to_numpy(dtype=float), df["target"].to_numpy(), list(features.columns)


def train_test_split_np(X, y, test_size=0.3, seed=42, stratify=True):
    """Simple NumPy train/test split (optionally stratified)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if stratify:
        test_idx = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            test_idx.extend(cls_idx[: int(round(len(cls_idx) * test_size))])
        test_idx = np.array(test_idx)
    else:
        perm = rng.permutation(n)
        test_idx = perm[: int(round(n * test_size))]
    mask = np.zeros(n, dtype=bool)
    mask[test_idx] = True
    return X[~mask], X[mask], y[~mask], y[mask]

"""Bootstrap Ensembles from Scratch: main entry point.

Run:  python main.py
Outputs figures to results/.
"""
from pathlib import Path

import numpy as np

from src import (
    bootstrap_sample,
    oob_indices,
    unique_fraction,
    simulate_unique_fraction,
    tree_variance_experiment,
    prediction_variance_per_point,
)
from utils import (
    load_dataset,
    train_test_split_np,
    plot_unique_fraction,
    plot_bootstrap_counts,
    plot_tree_variance,
)

SEED = 42
RESULTS = Path(__file__).parent / "results"


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main():
    X, y, _ = load_dataset("breast_cancer")
    X_train, X_test, y_train, y_test = train_test_split_np(X, y, test_size=0.3, seed=SEED)
    print(f"Dataset: breast_cancer | train={X_train.shape} test={X_test.shape}")

    # 1) One bootstrap sample, step by step
    section("1. Anatomy of a bootstrap sample")
    Xb, yb, idx = bootstrap_sample(X_train, y_train, rng=SEED)
    oob = oob_indices(idx, len(X_train))
    print(f"Original size:        {len(X_train)}")
    print(f"Bootstrap size:       {len(idx)}")
    print(f"Unique points drawn:  {len(np.unique(idx))} ({unique_fraction(idx, len(X_train)):.1%})")
    print(f"Out-of-bag points:    {len(oob)} ({len(oob) / len(X_train):.1%})")
    counts = np.bincount(idx, minlength=len(X_train))
    plot_bootstrap_counts(counts, RESULTS / "bootstrap_counts.png")

    # 2) The 63.2% rule
    section("2. The 63.2% rule (1 - 1/e)")
    ns = [10, 20, 50, 100, 200, 500, 1000, 5000]
    fracs = [simulate_unique_fraction(n, n_trials=200, rng=SEED) for n in ns]
    for n, f in zip(ns, fracs):
        print(f"  n={n:<5d} empirical={f:.4f}   theory={1 - 1 / np.e:.4f}")
    plot_unique_fraction(ns, fracs, RESULTS / "unique_fraction.png")

    # 3) Single trees are high-variance
    section("3. High variance of a single decision tree")
    out = tree_variance_experiment(X_train, y_train, X_test, y_test,
                                   n_trees=200, seed=SEED)
    accs = out["test_accs"]
    pv = prediction_variance_per_point(out["preds"])
    print(f"Tree on full train set: {out['full_acc']:.3f}")
    print(f"Bootstrap trees (n=200): mean={accs.mean():.3f}  std={accs.std():.3f}  "
          f"min={accs.min():.3f}  max={accs.max():.3f}")
    print(f"Test points where trees disagree (var > 0): {(pv > 0).mean():.1%}")
    print(f"Mean per-point prediction variance: {pv.mean():.4f}")
    plot_tree_variance(accs, out["full_acc"], RESULTS / "tree_variance.png")

    section("Done")
    print(f"Figures saved in: {RESULTS}/")
    print("Next (Day 2): average these unstable trees -> Bagging + OOB error.")


if __name__ == "__main__":
    main()

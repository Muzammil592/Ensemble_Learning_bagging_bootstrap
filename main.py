"""Bootstrap Ensembles from Scratch: main entry point.

Run:  python main.py
Outputs figures to results/.

Sections 1-3: bootstrap fundamentals and single-tree variance.
Sections 4-7: bagging from scratch, OOB error, error curves, regression.
"""
from pathlib import Path

import numpy as np
from sklearn.ensemble import BaggingClassifier as SkBaggingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src import (
    BaggingClassifier,
    BaggingRegressor,
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
    plot_error_vs_estimators,
    plot_oob_vs_test,
    plot_model_comparison,
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

    # ------------------------------------------------------------------
    # Day 2: Bagging from scratch + out-of-bag error
    # ------------------------------------------------------------------
    section("4. Bagging vs single tree vs scikit-learn (20 random splits)")
    n_estimators, n_splits = 50, 20
    single, ours, theirs, oob_list = [], [], [], []
    for s in range(n_splits):
        Xtr, Xte, ytr, yte = train_test_split_np(X, y, test_size=0.3, seed=s)
        single.append(DecisionTreeClassifier(random_state=s).fit(Xtr, ytr).score(Xte, yte))
        bag = BaggingClassifier(n_estimators=n_estimators, oob_score=True,
                                random_state=s).fit(Xtr, ytr)
        ours.append(bag.score(Xte, yte))
        oob_list.append(bag.oob_score_)
        sk = SkBaggingClassifier(DecisionTreeClassifier(), n_estimators=n_estimators,
                                 random_state=s).fit(Xtr, ytr)
        theirs.append(sk.score(Xte, yte))
    names = ["Single tree", "Bagging (ours)", "Bagging (sklearn)"]
    means = [np.mean(single), np.mean(ours), np.mean(theirs)]
    stds = [np.std(single), np.std(ours), np.std(theirs)]
    for n_, m_, s_ in zip(names, means, stds):
        print(f"  {n_:<18s} accuracy = {m_:.4f} +/- {s_:.4f}")
    print(f"  Variance reduction (std): {stds[0]:.4f} -> {stds[1]:.4f}")
    plot_model_comparison(names, means, stds, RESULTS / "model_comparison.png")

    section("5. Out-of-bag error: a free validation estimate")
    gap = np.abs(np.array(oob_list) - np.array(ours))
    print(f"  Mean OOB accuracy:  {np.mean(oob_list):.4f}")
    print(f"  Mean test accuracy: {np.mean(ours):.4f}")
    print(f"  Mean |OOB - test| gap: {gap.mean():.4f}  (max {gap.max():.4f})")
    plot_oob_vs_test(oob_list, ours, RESULTS / "oob_vs_test.png")

    section("6. Error vs number of trees (300 trees)")
    big = BaggingClassifier(n_estimators=300, oob_score=True,
                            random_state=SEED).fit(X_train, y_train)
    test_curve = [float((p != y_test).mean()) for p in big.staged_predict(X_test)]
    single_err = 1 - DecisionTreeClassifier(random_state=SEED).fit(X_train, y_train).score(X_test, y_test)
    for k in (1, 5, 10, 25, 50, 100, 300):
        print(f"  {k:>3d} trees: test error = {test_curve[k - 1]:.4f}"
              f" | OOB error = {big.oob_error_curve_[k - 1]:.4f}")
    print(f"  Single tree test error: {single_err:.4f}")
    plot_error_vs_estimators(big.oob_error_curve_, test_curve, single_err,
                             RESULTS / "error_vs_estimators.png")

    section("7. Bagging for regression (diabetes dataset)")
    Xr, yr, _ = load_dataset("diabetes")
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split_np(Xr, yr, test_size=0.3, seed=SEED,
                                                      stratify=False)
    tree_r2 = DecisionTreeRegressor(random_state=SEED).fit(Xr_tr, yr_tr).score(Xr_te, yr_te)
    reg = BaggingRegressor(n_estimators=100, oob_score=True,
                           random_state=SEED).fit(Xr_tr, yr_tr)
    print(f"  Single tree R^2:  {tree_r2:.3f}")
    print(f"  Bagging R^2:      {reg.score(Xr_te, yr_te):.3f}")
    print(f"  Bagging OOB R^2:  {reg.oob_score_:.3f}  (OOB MSE = {reg.oob_mse_:.1f})")

    section("Done")
    print(f"Figures saved in: {RESULTS}/")
    print("Next (Day 3): random feature subsetting -> Random Forest + bias-variance analysis.")


if __name__ == "__main__":
    main()

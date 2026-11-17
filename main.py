"""Bootstrap Ensembles from Scratch: main entry point.

Run:  python main.py
Outputs figures to results/.

Sections 1-3: bootstrap fundamentals and single-tree variance.
Sections 4-7: bagging from scratch, OOB error, error curves, regression.
Sections 8-12: random forest, bias-variance decomposition, max_features sweep,
               bagging vs random forest error curves, OOB vs test, feature importance.
"""
from pathlib import Path

import numpy as np
from sklearn.ensemble import BaggingClassifier as SkBaggingClassifier
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src import (
    BaggingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    bias_variance_decomposition,
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
    plot_grouped_bars,
    plot_bias_variance,
    plot_max_features_sweep,
    plot_error_curves,
    plot_feature_importance,
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

    # ------------------------------------------------------------------
    # Day 3: Random forest + bias-variance analysis
    # ------------------------------------------------------------------
    section("8. Single tree vs bagging vs random forest (20 splits, 100 trees)")
    datasets = ["breast_cancer", "wine"]
    model_names = ["Single tree", "Bagging (ours)", "Random forest (ours)", "Random forest (sklearn)"]
    table = {n: ([], []) for n in model_names}
    for ds in datasets:
        Xd, yd, _ = load_dataset(ds)
        res = {n: [] for n in model_names}
        for sd in range(20):
            Xtr, Xte, ytr, yte = train_test_split_np(Xd, yd, test_size=0.3, seed=sd)
            res["Single tree"].append(DecisionTreeClassifier(random_state=sd).fit(Xtr, ytr).score(Xte, yte))
            res["Bagging (ours)"].append(BaggingClassifier(n_estimators=100, voting="soft", random_state=sd).fit(Xtr, ytr).score(Xte, yte))
            res["Random forest (ours)"].append(RandomForestClassifier(n_estimators=100, random_state=sd).fit(Xtr, ytr).score(Xte, yte))
            res["Random forest (sklearn)"].append(SkRandomForestClassifier(n_estimators=100, random_state=sd).fit(Xtr, ytr).score(Xte, yte))
        print(f"  [{ds}]")
        for n in model_names:
            table[n][0].append(np.mean(res[n]))
            table[n][1].append(np.std(res[n]))
            print(f"    {n:<25s} {np.mean(res[n]):.4f} +/- {np.std(res[n]):.4f}")
    plot_grouped_bars(["Breast Cancer", "Wine"], table, RESULTS / "rf_comparison.png",
                      title="Test accuracy over 20 random splits (100 trees)")

    section("9. Bias-variance decomposition (Friedman #1, 100 training sets)")
    BV = dict(n_rounds=100, n_train=200, n_test=500, noise=1.0, seed=SEED)
    models = {
        "Single tree": lambda s_: DecisionTreeRegressor(random_state=s_),
        "Bagging": lambda s_: BaggingRegressor(n_estimators=50, random_state=s_),
        "Random forest": lambda s_: RandomForestRegressor(n_estimators=50, max_features=3, random_state=s_),
    }
    bv = {name: bias_variance_decomposition(f, **BV) for name, f in models.items()}
    print(f"  {'Model':<15s} {'Bias^2':>8s} {'Variance':>9s} {'Bias^2+Var':>11s}")
    for name, r in bv.items():
        print(f"  {name:<15s} {r['bias2']:>8.3f} {r['variance']:>9.3f} {r['mse_vs_truth']:>11.3f}")
    print(f"  Irreducible noise (sigma^2): {BV['noise'] ** 2:.1f}")
    plot_bias_variance(list(bv), [r["bias2"] for r in bv.values()],
                       [r["variance"] for r in bv.values()], BV["noise"] ** 2,
                       RESULTS / "bias_variance.png")

    section("10. Random forest: sweeping max_features (10 features total)")
    mf_values = [1, 2, 3, 5, 7, 10]
    sweep = []
    for mf in mf_values:
        r = bias_variance_decomposition(
            lambda s_, mf=mf: RandomForestRegressor(n_estimators=50, max_features=mf, random_state=s_), **BV)
        sweep.append(r)
        print(f"  max_features={mf:<2d} bias^2={r['bias2']:.3f} variance={r['variance']:.3f} total={r['mse_vs_truth']:.3f}")
    plot_max_features_sweep(mf_values, [r["bias2"] for r in sweep],
                            [r["variance"] for r in sweep], RESULTS / "max_features_sweep.png")

    section("11. Error vs number of trees: bagging vs random forest (10 splits)")
    K, n_rep = 200, 10
    bag_test, rf_test, bag_oob, rf_oob = [], [], [], []
    for sd in range(n_rep):
        Xtr, Xte, ytr, yte = train_test_split_np(X, y, test_size=0.3, seed=sd)
        bag = BaggingClassifier(n_estimators=K, voting="soft", oob_score=True, random_state=sd).fit(Xtr, ytr)
        rf = RandomForestClassifier(n_estimators=K, oob_score=True, random_state=sd).fit(Xtr, ytr)
        bag_test.append([float((p != yte).mean()) for p in bag.staged_predict(Xte)])
        rf_test.append([float((p != yte).mean()) for p in rf.staged_predict(Xte)])
        bag_oob.append(bag.oob_error_curve_)
        rf_oob.append(rf.oob_error_curve_)
    bag_test, rf_test, bag_oob, rf_oob = map(np.array, (bag_test, rf_test, bag_oob, rf_oob))
    ks = np.arange(1, K + 1)
    for k in (1, 5, 10, 25, 50, 100, 200):
        print(f"  {k:>3d} trees: bagging test err = {bag_test[:, k - 1].mean():.4f} | "
              f"RF test err = {rf_test[:, k - 1].mean():.4f}")
    diff = bag_test[:, -1] - rf_test[:, -1]      # paired: same split, same test set
    print(f"  Paired difference at {K} trees (bagging - RF): mean = {diff.mean():+.4f}, "
          f"RF better/equal in {(diff >= 0).sum()}/{n_rep} splits")
    plot_error_curves(ks, [
        dict(label="Bagging", mean=bag_test.mean(0), std=bag_test.std(0), color="#d9534f"),
        dict(label="Random forest", mean=rf_test.mean(0), std=rf_test.std(0), color="#4a90d9"),
    ], RESULTS / "bagging_vs_rf_curve.png",
        title="Test error vs number of trees (mean +/- std over 10 splits)")

    section("12. OOB error vs test error, and feature importance")
    print(f"  Bagging  | test err = {bag_test[:, -1].mean():.4f} | OOB err = {bag_oob[:, -1].mean():.4f}")
    print(f"  RF       | test err = {rf_test[:, -1].mean():.4f} | OOB err = {rf_oob[:, -1].mean():.4f}")
    print(f"  Mean |OOB - test| at {K} trees: bagging = {np.abs(bag_oob[:, -1] - bag_test[:, -1]).mean():.4f}, "
          f"RF = {np.abs(rf_oob[:, -1] - rf_test[:, -1]).mean():.4f}")
    plot_error_curves(ks, [
        dict(label="RF OOB error", mean=rf_oob.mean(0), std=rf_oob.std(0), color="#e08e0b"),
        dict(label="RF test error", mean=rf_test.mean(0), std=rf_test.std(0), color="#4a90d9"),
    ], RESULTS / "rf_oob_vs_test_curve.png",
        title="Random forest: OOB error vs test error (mean +/- std over 10 splits)")

    _, _, feat_names = load_dataset("breast_cancer")
    rf_full = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(X_train, y_train)
    imp = rf_full.feature_importances_
    top = np.argsort(imp)[::-1][:5]
    print("  Top 5 features: " + ", ".join(f"{feat_names[i]} ({imp[i]:.3f})" for i in top))
    plot_feature_importance(feat_names, imp, RESULTS / "feature_importance.png")

    section("Done")
    print(f"Figures saved in: {RESULTS}/")
    print("Next (Day 4): bootstrap confidence intervals + benchmarks.")


if __name__ == "__main__":
    main()

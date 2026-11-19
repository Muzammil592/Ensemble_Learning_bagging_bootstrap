"""Bootstrap Ensembles from Scratch: main entry point.

Run:  python main.py
Outputs figures to results/.

Sections 1-3: bootstrap fundamentals and single-tree variance.
Sections 4-7: bagging from scratch, OOB error, error curves, regression.
Sections 8-12: random forest, bias-variance decomposition, max_features sweep,
               bagging vs random forest error curves, OOB vs test, feature importance.
Sections 13-17: bootstrap confidence intervals, CI coverage check, cross-validated
               benchmarks with CIs, prediction intervals, bootstrap vs pasting vs Bayesian.
"""
from pathlib import Path

import numpy as np
from sklearn.ensemble import BaggingClassifier as SkBaggingClassifier
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.ensemble import BaggingRegressor as SkBaggingRegressor
from sklearn.ensemble import RandomForestRegressor as SkRandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src import (
    BaggingClassifier,
    BaggingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    BayesianBaggingClassifier,
    bias_variance_decomposition,
    make_friedman_data,
    accuracy,
    rmse,
    r2,
    bootstrap_mean_ci,
    bootstrap_ci,
    paired_bootstrap_diff_ci,
    coverage_experiment,
    tree_spread_interval,
    oob_residual_interval,
    interval_coverage,
    cv_predict,
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
    plot_bootstrap_distribution,
    plot_ci_coverage,
    plot_ci_forest,
    plot_interval_example,
    plot_interval_coverage,
)

SEED = 42
RESULTS = Path(__file__).parent / "results"


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def run_day4():
    """Day 4: bootstrap for uncertainty, cross-validated benchmarks, sampling schemes."""

    # ------------------------------------------------------------------
    section("13. Bootstrap confidence intervals for test metrics")
    Xc, yc, _ = load_dataset("breast_cancer")
    Xtr, Xte, ytr, yte = train_test_split_np(Xc, yc, test_size=0.3, seed=SEED)
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED).fit(Xtr, ytr)
    correct = (rf.predict(Xte) == yte).astype(float)
    ci = bootstrap_mean_ci(correct, n_boot=5000, rng=SEED)
    p, n_te = ci["estimate"], len(yte)
    print(f"  Breast Cancer RF accuracy on {n_te} test points: {p:.4f}")
    print(f"  95% bootstrap CI: [{ci['lo']:.4f}, {ci['hi']:.4f}]   (bootstrap SE = {ci['se']:.4f})")
    print(f"  Analytic SE sqrt(p(1-p)/n) for comparison: {np.sqrt(p * (1 - p) / n_te):.4f}")
    plot_bootstrap_distribution(ci["boot"], p, ci["lo"], ci["hi"],
                                RESULTS / "bootstrap_accuracy_ci.png",
                                title=f"Bootstrap distribution of test accuracy (n = {n_te})")

    Xr, yr, _ = load_dataset("diabetes")
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split_np(Xr, yr, test_size=0.3, seed=SEED, stratify=False)
    rfr = RandomForestRegressor(n_estimators=200, random_state=SEED).fit(Xr_tr, yr_tr)
    pr = rfr.predict(Xr_te)
    for name, fn in (("RMSE", rmse), ("R^2", r2)):
        c = bootstrap_ci(yr_te, pr, fn, n_boot=2000, rng=SEED)
        print(f"  Diabetes RF {name} on {len(yr_te)} test points: {c['estimate']:.3f}  "
              f"95% CI [{c['lo']:.3f}, {c['hi']:.3f}]")

    # ------------------------------------------------------------------
    section("14. Do the 95% intervals really contain the truth 95% of the time?")
    Xd, yd, _ = load_dataset("digits")
    Xd_tr, Xd_pool, yd_tr, yd_pool = train_test_split_np(Xd, yd, test_size=0.7, seed=SEED)
    rf_d = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(Xd_tr, yd_tr)
    pool_correct = (rf_d.predict(Xd_pool) == yd_pool).astype(float)
    sizes = [20, 50, 100, 200, 500]
    truth, cov = coverage_experiment(pool_correct, sizes, n_trials=1000, n_boot=1000, seed=SEED)
    print(f"  True accuracy of the fixed model on the pool ({len(pool_correct)} digits): {truth:.4f}")
    for n_ in sizes:
        print(f"  test size n={n_:<4d} coverage = {cov[n_]:.3f}  (nominal 0.950)")
    plot_ci_coverage(sizes, [cov[n_] for n_ in sizes], 0.95, RESULTS / "ci_coverage.png", 1000)

    # ------------------------------------------------------------------
    section("15. Benchmarks: 5-fold CV predictions + bootstrap CIs")
    cls_models = {
        "Single tree": lambda s_: DecisionTreeClassifier(random_state=s_),
        "Bagging (ours)": lambda s_: BaggingClassifier(n_estimators=100, voting="soft", random_state=s_),
        "RF (ours)": lambda s_: RandomForestClassifier(n_estimators=100, random_state=s_),
        "Bagging (sklearn)": lambda s_: SkBaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=s_),
        "RF (sklearn)": lambda s_: SkRandomForestClassifier(n_estimators=100, random_state=s_),
    }
    reg_models = {
        "Single tree": lambda s_: DecisionTreeRegressor(random_state=s_),
        "Bagging (ours)": lambda s_: BaggingRegressor(n_estimators=100, random_state=s_),
        "RF (ours)": lambda s_: RandomForestRegressor(n_estimators=100, random_state=s_),
        "Bagging (sklearn)": lambda s_: SkBaggingRegressor(n_estimators=100, random_state=s_),
        "RF (sklearn)": lambda s_: SkRandomForestRegressor(n_estimators=100, max_features=1 / 3, random_state=s_),
    }
    comparisons = [("RF (ours)", "Bagging (ours)"), ("RF (ours)", "RF (sklearn)"),
                   ("Bagging (ours)", "Bagging (sklearn)")]

    cls_panels = {}
    for ds in ["iris", "wine", "breast_cancer", "digits"]:
        Xb, yb, _ = load_dataset(ds)
        preds = {name: cv_predict(f, Xb, yb, k=5, seed=SEED) for name, f in cls_models.items()}
        print(f"  [{ds}]  n={len(yb)}  (accuracy, 95% CI)")
        cls_panels[f"{ds} (n={len(yb)})"] = {}
        for name, pr_ in preds.items():
            c = bootstrap_mean_ci((pr_ == yb).astype(float), n_boot=2000, rng=SEED)
            cls_panels[f"{ds} (n={len(yb)})"][name] = (c["estimate"], c["lo"], c["hi"])
            print(f"    {name:<19s} {c['estimate']:.4f}  [{c['lo']:.4f}, {c['hi']:.4f}]")
        for a, b in comparisons:
            d = paired_bootstrap_diff_ci(yb, preds[a], preds[b], accuracy, n_boot=2000, rng=SEED)
            flag = "excludes 0" if d["excludes_zero"] else "includes 0"
            print(f"      paired diff {a} - {b}: {d['estimate']:+.4f}  [{d['lo']:+.4f}, {d['hi']:+.4f}]  ({flag})")
    plot_ci_forest(cls_panels, RESULTS / "benchmark_classification_ci.png",
                   xlabel="5-fold CV accuracy (95% bootstrap CI)",
                   suptitle="Classification benchmark")

    reg_panels = {}
    for ds in ["diabetes", "friedman1"]:
        Xb, yb, _ = load_dataset(ds)
        preds = {name: cv_predict(f, Xb, yb, k=5, seed=SEED, stratified=False)
                 for name, f in reg_models.items()}
        print(f"  [{ds}]  n={len(yb)}  (RMSE, 95% CI; lower is better)")
        reg_panels[f"{ds} (n={len(yb)})"] = {}
        for name, pr_ in preds.items():
            c = bootstrap_ci(yb, pr_, rmse, n_boot=1000, rng=SEED)
            reg_panels[f"{ds} (n={len(yb)})"][name] = (c["estimate"], c["lo"], c["hi"])
            print(f"    {name:<19s} {c['estimate']:.3f}  [{c['lo']:.3f}, {c['hi']:.3f}]")
        for a, b in comparisons:
            d = paired_bootstrap_diff_ci(yb, preds[a], preds[b], rmse, n_boot=1000, rng=SEED)
            flag = "excludes 0" if d["excludes_zero"] else "includes 0"
            print(f"      paired diff {a} - {b}: {d['estimate']:+.3f}  [{d['lo']:+.3f}, {d['hi']:+.3f}]  ({flag})")
    plot_ci_forest(reg_panels, RESULTS / "benchmark_regression_ci.png",
                   xlabel="5-fold CV RMSE (95% bootstrap CI, lower is better)",
                   suptitle="Regression benchmark")

    # ------------------------------------------------------------------
    section("16. Prediction intervals from an ensemble (95% nominal)")
    cov_spread, cov_oob, w_spread, w_oob = {}, {}, {}, {}
    for ds in ["diabetes", "friedman1"]:
        Xb, yb, _ = load_dataset(ds)
        cs, co, ws, wo = [], [], [], []
        for sd in range(20):
            Xtr, Xte, ytr, yte = train_test_split_np(Xb, yb, test_size=0.3, seed=sd, stratify=False)
            m = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=sd).fit(Xtr, ytr)
            c1, w1 = interval_coverage(yte, *tree_spread_interval(m, Xte))
            c2, w2 = interval_coverage(yte, *oob_residual_interval(m, Xte, ytr))
            cs.append(c1); ws.append(w1); co.append(c2); wo.append(w2)
        cov_spread[ds], cov_oob[ds], w_spread[ds], w_oob[ds] = map(np.mean, (cs, co, ws, wo))
        print(f"  [{ds}] tree-spread: coverage={cov_spread[ds]:.3f} width={w_spread[ds]:.1f} | "
              f"OOB-residual: coverage={cov_oob[ds]:.3f} width={w_oob[ds]:.1f}")
    plot_interval_coverage(["diabetes", "friedman1"],
                           {"Tree spread": [cov_spread[d] for d in ("diabetes", "friedman1")],
                            "OOB residuals": [cov_oob[d] for d in ("diabetes", "friedman1")]},
                           0.95, RESULTS / "interval_coverage.png")

    print("  Noise sweep on Friedman #1 (500 train / 1000 test points):")
    for noise in (0.5, 1, 2, 4, 8):
        Xtr, ytr = make_friedman_data(500, rng=10, noise=noise)
        Xte, yte = make_friedman_data(1000, rng=11, noise=noise)
        m = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=SEED).fit(Xtr, ytr)
        c1, w1 = interval_coverage(yte, *tree_spread_interval(m, Xte))
        c2, w2 = interval_coverage(yte, *oob_residual_interval(m, Xte, ytr))
        print(f"    noise={noise:<4} tree-spread cov={c1:.3f} w={w1:5.1f} | OOB-residual cov={c2:.3f} w={w2:5.1f}")

    Xf, yf, _ = load_dataset("friedman1")
    Xtr, Xte, ytr, yte = train_test_split_np(Xf, yf, test_size=0.3, seed=0, stratify=False)
    m = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=0).fit(Xtr, ytr)
    lo1, hi1 = tree_spread_interval(m, Xte)
    lo2, hi2 = oob_residual_interval(m, Xte, ytr)
    plot_interval_example(yte, m.predict(Xte), [lo1, lo2], [hi1, hi2],
                          ["Tree-spread interval", "OOB-residual interval"],
                          RESULTS / "prediction_intervals.png")

    # ------------------------------------------------------------------
    section("17. Bonus: bootstrap vs pasting vs Bayesian bootstrap (5-fold CV)")
    schemes = {
        "Bootstrap": lambda s_: BaggingClassifier(n_estimators=100, voting="soft", random_state=s_),
        "Pasting 63.2%": lambda s_: BaggingClassifier(n_estimators=100, voting="soft", bootstrap=False, max_samples=0.632, random_state=s_),
        "Pasting 80%": lambda s_: BaggingClassifier(n_estimators=100, voting="soft", bootstrap=False, max_samples=0.8, random_state=s_),
        "Bayesian bootstrap": lambda s_: BayesianBaggingClassifier(n_estimators=100, voting="soft", random_state=s_),
    }
    ds_list = ["iris", "wine", "breast_cancer", "digits"]
    means = {k: [] for k in schemes}
    ses = {k: [] for k in schemes}
    for ds in ds_list:
        Xb, yb, _ = load_dataset(ds)
        preds = {k: cv_predict(f, Xb, yb, k=5, seed=SEED) for k, f in schemes.items()}
        print(f"  [{ds}]")
        for k, pr_ in preds.items():
            c = bootstrap_mean_ci((pr_ == yb).astype(float), n_boot=2000, rng=SEED)
            means[k].append(c["estimate"]); ses[k].append(c["se"])
            line = f"    {k:<19s} {c['estimate']:.4f}  [{c['lo']:.4f}, {c['hi']:.4f}]"
            if k != "Bootstrap":
                d = paired_bootstrap_diff_ci(yb, pr_, preds["Bootstrap"], accuracy, n_boot=1000, rng=SEED)
                line += f"   vs bootstrap: {d['estimate']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
            print(line)
    plot_grouped_bars(["iris", "wine", "breast cancer", "digits"],
                      {k: (means[k], ses[k]) for k in schemes},
                      RESULTS / "sampling_schemes.png", ylabel="5-fold CV accuracy",
                      title="Sampling schemes for the ensemble (bars: bootstrap SE)")


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

    run_day4()

    section("Done")
    print(f"Figures saved in: {RESULTS}/")
    print("Next (Day 5): polish, CI workflow, final write-up.")


if __name__ == "__main__":
    main()

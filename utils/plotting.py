"""Plotting helpers. All functions save a figure and return its path."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_unique_fraction(ns, fractions, path):
    """Unique-sample fraction vs dataset size, with the 1 - 1/e limit."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, fractions, "o-", label="Empirical (mean of simulations)")
    ax.axhline(1 - 1 / np.e, color="crimson", ls="--",
               label=r"Theory: $1 - 1/e \approx 0.632$")
    ax.set_xscale("log")
    ax.set_xlabel("Dataset size n")
    ax.set_ylabel("Fraction of unique points in bootstrap sample")
    ax.set_title("A bootstrap sample contains ~63.2% of the original data")
    ax.legend()
    return _save(fig, path)


def plot_bootstrap_counts(counts, path, n_show=60):
    """How many times each of the first n_show points was drawn in one bootstrap."""
    counts = np.asarray(counts)[:n_show]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    colors = ["#d9534f" if c == 0 else "#4a90d9" for c in counts]
    ax.bar(np.arange(len(counts)), counts, color=colors)
    ax.set_xlabel("Original sample index")
    ax.set_ylabel("Times drawn")
    ax.set_title("One bootstrap sample: red = out-of-bag, blue = drawn 1+ times", fontsize=10)
    return _save(fig, path)


def plot_tree_variance(test_accs, full_acc, path):
    """Distribution of test accuracy of trees trained on different bootstrap samples."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(test_accs, bins=15, color="#4a90d9", edgecolor="white")
    ax.axvline(np.mean(test_accs), color="black", ls="--",
               label=f"Mean = {np.mean(test_accs):.3f}")
    ax.axvline(full_acc, color="crimson", ls="-",
               label=f"Tree on full train set = {full_acc:.3f}")
    ax.set_xlabel("Test accuracy")
    ax.set_ylabel("Number of trees")
    ax.set_title(f"Single trees are unstable (std = {np.std(test_accs):.3f})")
    ax.legend()
    return _save(fig, path)


# ----------------------------- Day 2 plots -----------------------------

def plot_error_vs_estimators(oob_curve, test_curve, single_tree_err, path,
                             ylabel="Error rate"):
    """OOB and test error as the ensemble grows, vs. a single tree."""
    ks = np.arange(1, len(test_curve) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    if len(oob_curve):
        # the OOB curve can be shorter at the start; align it to the right
        ax.plot(ks[-len(oob_curve):], oob_curve, label="OOB error", color="#e08e0b")
    ax.plot(ks, test_curve, label="Test error", color="#4a90d9")
    ax.axhline(single_tree_err, color="crimson", ls="--",
               label=f"Single tree ({single_tree_err:.3f})")
    ax.set_xlabel("Number of trees")
    ax.set_ylabel(ylabel)
    ax.set_title("Error vs number of trees (single split, seed 42)")
    ax.legend()
    return _save(fig, path)


def plot_oob_vs_test(oob_scores, test_scores, path):
    """Scatter of OOB accuracy vs test accuracy across many random splits."""
    oob_scores, test_scores = np.asarray(oob_scores), np.asarray(test_scores)
    lo = min(oob_scores.min(), test_scores.min()) - 0.01
    hi = max(oob_scores.max(), test_scores.max()) + 0.01
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(oob_scores, test_scores, color="#4a90d9", edgecolor="white", s=55)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="OOB = test")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("OOB accuracy (free, from training data)")
    ax.set_ylabel("Test accuracy (held-out data)")
    ax.set_title(f"OOB tracks test accuracy (mean gap = "
                 f"{np.mean(np.abs(oob_scores - test_scores)):.3f})", fontsize=10)
    ax.legend()
    return _save(fig, path)


def plot_model_comparison(names, means, stds, path, ylabel="Test accuracy"):
    """Bar chart comparing models over repeated splits (mean +/- std)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#d9534f", "#4a90d9", "#5cb85c", "#9b59b6"][: len(names)]
    ax.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor="white")
    lo = min(np.array(means) - np.array(stds))
    ax.set_ylim(max(0, lo - 0.03), min(1.0, max(np.array(means) + np.array(stds)) + 0.02)
                if ylabel.startswith("Test acc") else None)
    for i, m in enumerate(means):
        ax.text(i, m + stds[i] + 0.002, f"{m:.3f}", ha="center", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title("Single tree vs from-scratch bagging vs scikit-learn bagging")
    return _save(fig, path)


# ----------------------------- Day 3 plots -----------------------------

def plot_grouped_bars(group_labels, series, path, ylabel="Test accuracy", title=""):
    """Grouped bars. series = {name: (means, stds)}, one value per group."""
    names = list(series)
    x = np.arange(len(group_labels))
    w = 0.8 / len(names)
    colors = ["#d9534f", "#4a90d9", "#5cb85c", "#9b59b6", "#e08e0b"]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    all_vals = []
    for i, name in enumerate(names):
        means, stds = series[name]
        ax.bar(x + (i - (len(names) - 1) / 2) * w, means, w, yerr=stds, capsize=3,
               label=name, color=colors[i % len(colors)], edgecolor="white")
        all_vals += list(np.array(means) - np.array(stds))
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylim(max(0, min(all_vals) - 0.03), 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, path)


def plot_bias_variance(names, bias2, variance, noise, path):
    """Stacked bars: bias^2 + variance (+ irreducible noise as a dashed line)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(names, bias2, label="Bias²", color="#d9534f", edgecolor="white")
    ax.bar(names, variance, bottom=bias2, label="Variance", color="#4a90d9",
           edgecolor="white")
    for i in range(len(names)):
        ax.text(i, bias2[i] / 2, f"{bias2[i]:.2f}", ha="center", color="white", fontsize=9)
        ax.text(i, bias2[i] + variance[i] / 2, f"{variance[i]:.2f}", ha="center",
                color="white", fontsize=9)
        ax.text(i, bias2[i] + variance[i] + 0.15, f"{bias2[i] + variance[i]:.2f}",
                ha="center", fontsize=9)
    ax.axhline(noise, color="gray", ls="--", label=f"Irreducible noise = {noise:.1f}")
    ax.set_ylabel("Mean squared error")
    ax.set_title("Bias-variance decomposition (synthetic Friedman #1)")
    ax.legend()
    return _save(fig, path)


def plot_max_features_sweep(mf_values, bias2, variance, path, n_features=10):
    """Bias^2, variance and their sum as the number of features per split grows."""
    total = np.array(bias2) + np.array(variance)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(mf_values, bias2, "o-", color="#d9534f", label="Bias²")
    ax.plot(mf_values, variance, "s-", color="#4a90d9", label="Variance")
    ax.plot(mf_values, total, "^-", color="black", label="Bias² + Variance")
    best = int(np.argmin(total))
    ax.scatter([mf_values[best]], [total[best]], s=180, facecolors="none",
               edgecolors="green", linewidths=2, label=f"Best: {mf_values[best]} features")
    ax.set_xlabel(f"max_features (features tried per split, out of {n_features})")
    ax.set_ylabel("Mean squared error")
    ax.set_title("Random forest: trading variance against bias")
    ax.legend()
    return _save(fig, path)


def plot_error_curves(ks, series, path, title="", ylabel="Test error", hlines=None):
    """Generic error-vs-trees plot. series = list of dicts(label, mean, std, color, ls)."""
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for s in series:
        ax.plot(ks, s["mean"], label=s["label"], color=s.get("color"),
                ls=s.get("ls", "-"))
        if s.get("std") is not None:
            ax.fill_between(ks, np.array(s["mean"]) - s["std"],
                            np.array(s["mean"]) + s["std"],
                            color=s.get("color"), alpha=0.15)
    for label, val, color in (hlines or []):
        ax.axhline(val, color=color, ls=":", label=label)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.legend()
    return _save(fig, path)


def plot_feature_importance(names, importances, path, top=10):
    """Horizontal bar chart of the top-N feature importances."""
    order = np.argsort(importances)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(np.array(names)[order], np.array(importances)[order], color="#4a90d9")
    ax.set_xlabel("Mean importance across trees")
    ax.set_title(f"Random forest: top {top} features (Breast Cancer)")
    return _save(fig, path)

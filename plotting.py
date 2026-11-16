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

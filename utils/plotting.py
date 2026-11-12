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

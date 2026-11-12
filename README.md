# Bootstrap Ensembles from Scratch

Building bagging and random forests from scratch with NumPy, then studying *why*
bootstrapping reduces variance.

## Project structure

```
bootstrap-ensembles/
├── main.py            # entry point: python main.py
├── data/              # CSV datasets (breast_cancer, wine)
├── src/               # core algorithms (bootstrap, variance experiments)
├── utils/             # data loading and plotting helpers
├── results/           # generated figures
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

## Recent key findings

**1. A bootstrap sample contains ~63.2% of the original points.**
Each point is missed with probability (1 − 1/n)ⁿ → 1/e ≈ 0.368. Simulations
over dataset sizes from 10 to 5,000 converge to the theoretical value (0.6326 at n=1000).
The ~36.8% left out are the **out-of-bag (OOB)** points, a free validation set for Day 2.

**2. A single decision tree is unstable.**
On the Breast Cancer dataset, 200 trees trained on different bootstrap resamples of the
same training set had test accuracy ranging from **0.877 to 0.971** (std = 0.019), and
the trees disagreed on **43.9% of test points**. This instability (high variance) is exactly what
bagging averages away.


## What I learned

- Bootstrapping is sampling with replacement, and the 63.2% / 36.8% split is a direct
  consequence of the probability that a point is never drawn.
- Unpruned trees fit noise, so small changes in training data change predictions a lot.
- Averaging many decorrelated, high-variance models is the core idea behind bagging.

## Module reference

| Function | Location | Purpose |
|----------|----------|---------|
| `bootstrap_indices` | `src/bootstrap.py` | Draw indices with replacement |
| `bootstrap_sample` | `src/bootstrap.py` | Return resampled `X, y` and indices |
| `oob_indices` | `src/bootstrap.py` | Points never drawn in a sample |
| `simulate_unique_fraction` | `src/bootstrap.py` | Empirical check of the 63.2% rule |
| `tree_variance_experiment` | `src/variance_demo.py` | Train trees on resamples, record accuracy and predictions |
| `load_dataset` | `utils/data_loader.py` | Load a CSV from `data/` |

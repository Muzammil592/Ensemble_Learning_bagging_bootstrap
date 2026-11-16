# Bootstrap Ensembles from Scratch

Building bagging and random forests from scratch with NumPy, then studying *why*
bootstrapping reduces variance.

## Project structure

```
bootstrap-ensembles/
├── main.py            # entry point: python main.py
├── data/              # CSV datasets (breast_cancer, wine, diabetes)
├── src/               # core algorithms (bootstrap, bagging, variance experiments)
├── utils/             # data loading and plotting helpers
├── tests/             # pytest unit tests
├── results/           # generated figures
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python main.py        # runs all experiments, saves figures to results/
pytest -q             # runs the unit tests
```

## Progress

- [x] **Day 1:** Bootstrap fundamentals and the variance problem
- [x] **Day 2:** Bagging from scratch + out-of-bag error
- [ ] Day 3: Random forest + bias-variance analysis
- [ ] Day 4: Bootstrap confidence intervals + benchmarks
- [ ] Day 5: Polish, CI, write-up

## Day 1 key findings

**1. A bootstrap sample contains ~63.2% of the original points.**
Each point is missed with probability (1 - 1/n)^n -> 1/e ≈ 0.368. Simulations
from n=10 to n=5,000 converge to the theoretical value (0.6326 at n=1000).
The ~36.8% left out are the **out-of-bag (OOB)** points.

![unique fraction](results/unique_fraction.png)

**2. A single decision tree is unstable.**
200 trees trained on different bootstrap resamples of the same Breast Cancer training
set had test accuracy from **0.877 to 0.971** (std 0.019), and disagreed on
**43.9% of test points**.

![tree variance](results/tree_variance.png)

## Day 2 key findings

I implemented `BaggingClassifier` and `BaggingRegressor` in `src/bagging.py`. Only the base
tree comes from scikit-learn; resampling, hard/soft voting, averaging, OOB estimation and
staged predictions are my own code. Setting `bootstrap=False` gives *pasting*
(sampling without replacement).

**1. Bagging beats a single tree and matches scikit-learn.**
Averaged over 20 random train/test splits on Breast Cancer (50 trees):

| Model | Test accuracy (mean ± std) |
|-------|---------------------------|
| Single decision tree | 0.9243 ± 0.0182 |
| **Bagging (from scratch)** | **0.9535 ± 0.0140** |
| Bagging (scikit-learn) | 0.9526 ± 0.0163 |

Mine is within 0.001 of scikit-learn's, and the spread across splits shrinks (0.018 -> 0.014).

![model comparison](results/model_comparison.png)

**2. OOB accuracy is a free validation estimate.**
Each tree never sees ~36.8% of the training points, so those points can score it. Across the
same 20 splits, mean OOB accuracy was 0.9486 vs 0.9535 on a held-out test set (mean gap
0.015, max 0.040), without needing a validation split.

![oob vs test](results/oob_vs_test.png)

**3. Most of the gain arrives within the first ~25 trees.**
On one split (seed 42) the test error fell from 0.070 with 1 tree to 0.023 by 50 trees and then
stayed flat. On that split the test error is lower than OOB error (0.053), a reminder that a
single split is noisy; the 20-split average above is the fair comparison.

![error vs estimators](results/error_vs_estimators.png)

**4. Regression (diabetes, 100 trees).**
A single tree overfits badly (test R² = -0.383); bagging reaches R² = 0.386, with OOB R² = 0.475.
The OOB estimate is optimistic here, so on small datasets treat it as a rough guide.

## What I learned

- The 63.2% / 36.8% split follows directly from the probability a point is never drawn.
- Unpruned trees fit noise, so small data changes flip many predictions.
- Averaging bootstrap-trained trees cuts variance without any tuning, and OOB error comes for free.
- A single train/test split can mislead; repeated splits give a more honest comparison.

## Module reference

| Item | Location | Purpose |
|------|----------|---------|
| `bootstrap_sample`, `oob_indices` | `src/bootstrap.py` | Resampling and OOB indices |
| `BaggingClassifier` | `src/bagging.py` | Hard/soft voting, OOB score, staged predictions |
| `BaggingRegressor` | `src/bagging.py` | Mean aggregation, OOB R² and MSE |
| `tree_variance_experiment` | `src/variance_demo.py` | Single-tree variance study |
| `load_dataset` | `utils/data_loader.py` | Load a CSV from `data/` |
| `tests/test_bagging.py` | `tests/` | 16 tests incl. comparison with scikit-learn |

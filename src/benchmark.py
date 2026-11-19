"""Cross-validated out-of-fold predictions for fair benchmarking."""
import numpy as np

from utils.data_loader import kfold_indices


def cv_predict(model_factory, X, y, k=5, seed=0, stratified=True):
    """Return one out-of-fold prediction per sample.

    Every sample is predicted by a model that never saw it, so pooling the predictions
    lets us bootstrap over ALL samples to get confidence intervals.
    model_factory(seed) must return an unfitted model with fit / predict.
    """
    X, y = np.asarray(X), np.asarray(y)
    folds = kfold_indices(len(y), k=k, seed=seed, y=y if stratified else None)
    pred = np.empty(len(y), dtype=y.dtype)
    for f, (tr, te) in enumerate(folds):
        model = model_factory(seed + f)
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
    return pred

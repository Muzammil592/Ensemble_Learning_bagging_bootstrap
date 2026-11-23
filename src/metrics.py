"""Plain NumPy metrics."""
import numpy as np


def accuracy(y_true, y_pred):
    """Fraction of predictions that exactly match the labels."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def mse(y_true, y_pred):
    """Mean squared error."""
    d = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.mean(d ** 2))


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def r2(y_true, y_pred):
    """Coefficient of determination (R^2)."""
    y_true = np.asarray(y_true, float)
    ss_res = np.sum((y_true - np.asarray(y_pred, float)) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)

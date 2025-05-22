import numpy as np
import scipy


def plcc(x, y):
    return scipy.stats.pearsonr(x, y)[0]


def srcc(x, y):
    return scipy.stats.spearmanr(x, y)[0]


def krcc(x, y):
    return scipy.stats.kendalltau(x, y)[0]


def mae(x, y):
    return np.mean(np.abs(x - y))


def rmse(x, y):
    return np.sqrt(np.mean(np.square(x - y)))


def compute_metrics(y_true, y_pred, verbose=True):
    PLCC = np.round(plcc(y_true, y_pred), 3)
    SRCC = np.round(srcc(y_true, y_pred), 3)
    KRCC = np.round(krcc(y_true, y_pred), 3)

    RMSE = np.round(rmse(y_true, y_pred), 3)
    MAE = np.round(mae(y_true, y_pred), 3)

    if verbose:
        print(f"PLCC: {PLCC}  "
              f"SRCC: {SRCC}  "
              f"KRCC: {KRCC}  "
              f"RMSE: {RMSE}  "
              f"MAE: {MAE}")

    return PLCC, SRCC, KRCC, RMSE, MAE


def compute_mean_std(values, verbose=True):
    mean = np.mean(values)
    std = np.std(values)

    if verbose:
        print(f"mean: {mean:.4f}")
        print(f"std deviation: {std:.4f}")

    return mean, std

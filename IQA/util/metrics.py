import numpy as np
import scipy


# Pearson linear correlation coefficient
# It reflects the prediction linearity of the IQA algorithm.
def plcc(x, y):
    return scipy.stats.pearsonr(x, y)[0]


# Spearman rank-order correlation coefficient
# It indicates the prediction monotonicity.
def srcc(x, y):
    return scipy.stats.spearmanr(x, y)[0]


def mae(x, y):
    return np.mean(np.abs(x - y))


def rmse(x, y):
    return np.sqrt(np.mean(np.square(x - y)))


# Evaluation criteria
# PLCC, SRCC, MAE, RMSE values are reported during testing.
def evaluate_metrics(y_pred, y_true):
    PLCC = np.round(plcc(y_pred, y_true), 3)
    SRCC = np.round(srcc(y_pred, y_true), 3)

    MAE = np.round(mae(y_pred, y_true), 3)
    RMSE = np.round(rmse(y_pred, y_true), 3)

    return PLCC, SRCC, MAE, RMSE

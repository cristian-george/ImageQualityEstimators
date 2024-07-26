import numpy as np
import tensorflow as tf
from keras import backend as K
import scipy


# Pearson linear correlation coefficient
# It reflects the prediction linearity of the IQA algorithm.
def plcc(x, y):
    return scipy.stats.pearsonr(x, y)[0]


# Spearman rank-order correlation coefficient
# It indicates the prediction monotonicity.
def srcc(x, y):
    return scipy.stats.spearmanr(x, y)[0]


# Metrics used for tensorflow ops
def rmse_tf(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def plcc_tf(y_true, y_pred):
    centered_true = y_true - K.mean(y_true)
    centered_pred = y_pred - K.mean(y_pred)
    return K.mean(centered_true * centered_pred) / (K.std(y_true) * K.std(y_pred) + K.epsilon())


def srcc_tf(y_true, y_pred):
    srcc_value = tf.py_function(srcc, [y_true, y_pred], tf.float32)
    return srcc_value + K.epsilon()


# Evaluation criteria
# PLCC, SRCC, MAE, RMSE values are reported during testing.
def evaluate_metrics(y_pred, y_true):
    PLCC = np.round(plcc(y_pred, y_true), 3)
    SRCC = np.round(srcc(y_pred, y_true), 3)

    MAE = np.round(np.mean(np.abs(y_pred - y_true)), 3)
    RMSE = np.round(np.sqrt(np.mean((y_pred - y_true) ** 2)), 3)

    print("PLCC, SRCC, MAE, RMSE: ", PLCC, SRCC, MAE, RMSE)

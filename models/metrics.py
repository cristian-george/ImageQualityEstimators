import tensorflow as tf
from keras import backend as K
from scipy.stats import spearmanr, pearsonr


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def emd(y_true, y_pred):
    cdf_true = K.cumsum(y_true, axis=-1)
    cdf_pred = K.cumsum(y_pred, axis=-1)
    emd = K.sqrt(K.mean(K.square(cdf_true - cdf_pred), axis=-1))
    return K.mean(emd)


def srcc(y_true, y_pred):
    def compute_srcc(y_true, y_pred):
        return spearmanr(y_true, y_pred)[0]

    srcc_value = tf.py_function(compute_srcc, [y_true, y_pred], tf.float32)
    return srcc_value


# def plcc(y_true, y_pred):
#     def compute_plcc(y_true, y_pred):
#         return pearsonr(y_true.numpy().flatten(), y_pred.numpy().flatten())[0]
#
#     plcc_value = tf.py_function(compute_plcc, [y_true, y_pred], tf.float32)
#     return plcc_value

def plcc(y_true, y_pred):
    mean_true = K.mean(y_true)
    mean_pred = K.mean(y_pred)
    centered_true = y_true - mean_true
    centered_pred = y_pred - mean_pred
    numerator = K.sum(centered_true * centered_pred)
    denominator = K.sqrt(K.sum(K.square(centered_true)) * K.sum(K.square(centered_pred)))
    plcc_value = numerator / denominator
    return plcc_value

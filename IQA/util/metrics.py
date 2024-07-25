import tensorflow as tf
from keras import backend as K
from scipy.stats import spearmanr


def rmse_tf(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def plcc_tf(y_true, y_pred):
    centered_true = y_true - K.mean(y_true)
    centered_pred = y_pred - K.mean(y_pred)
    return K.mean(centered_true * centered_pred) / (K.std(y_true) * K.std(y_pred) + K.epsilon())


def srcc_tf(y_true, y_pred):
    def compute_srcc(y_true, y_pred):
        return spearmanr(y_true, y_pred)[0]

    srcc_value = tf.py_function(compute_srcc, [y_true, y_pred], tf.float32)
    return srcc_value + K.epsilon()

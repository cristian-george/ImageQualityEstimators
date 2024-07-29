import numpy as np
import tensorflow as tf
from keras import backend as K


# Metrics used for tensorflow ops
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def plcc(y_true, y_pred):
    centered_true = y_true - K.mean(y_true)
    centered_pred = y_pred - K.mean(y_pred)
    return K.mean(centered_true * centered_pred) / (K.std(y_true) * K.std(y_pred) + K.epsilon())


@tf.function
def rank(x):
    n = tf.shape(x)[0]

    x_expanded_1 = tf.tile(tf.reshape(x, [1, -1]), [n, 1])
    x_expanded_2 = tf.tile(tf.reshape(x, [-1, 1]), [1, n])

    less_than = tf.reduce_sum(tf.cast(x_expanded_1 < x_expanded_2, tf.float32), axis=1)
    equal_to = tf.reduce_sum(tf.cast(x_expanded_1 == x_expanded_2, tf.float32), axis=1)

    rank_x = less_than + (equal_to - 1) * 0.5 + 1

    return rank_x


def srcc(y_true, y_pred):
    return plcc(rank(y_true), rank(y_pred))

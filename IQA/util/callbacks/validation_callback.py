import numpy as np
import tensorflow as tf
from tqdm import tqdm

from util.metrics import evaluate_metrics


class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, data, target_size, loss):
        super().__init__()
        self.data = data
        self.target_size = target_size
        self.loss = loss

    def __evaluate_metrics(self, logs, y_true, y_pred):
        logs['val_loss'] = self.loss(y_true, y_pred)

        PLCC, SRCC, MAE, RMSE = evaluate_metrics(y_true, y_pred)
        logs['val_plcc'] = PLCC
        logs['val_srcc'] = SRCC
        logs['val_mae'] = MAE
        logs['val_rmse'] = RMSE

    def on_epoch_end(self, epoch, logs=None):
        y_pred = []
        y_true = []

        for images, labels in tqdm(self.data, total=len(self.data), desc='Validation'):
            batch_size = tf.shape(images)[0]
            patches = tf.reshape(images, (-1,) + self.target_size)
            patch_predictions = self.model.predict(patches, batch_size=batch_size, verbose=0)
            patch_predictions = tf.reshape(patch_predictions, (batch_size, 5))
            patch_predictions = tf.reduce_mean(patch_predictions, axis=1)
            y_pred.append(patch_predictions.numpy())
            y_true.append(labels.numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        self.__evaluate_metrics(logs, y_true, y_pred)

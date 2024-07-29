import numpy as np
import tensorflow as tf
from tqdm import tqdm

from util.metrics import evaluate_metrics


class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, data, loss, target_size):
        super().__init__()
        self.data = data
        self.loss = loss
        self.target_size = target_size

    def on_epoch_end(self, epoch, logs=None):
        y_pred = []
        y_true = []

        for images, labels in tqdm(self.data, total=len(self.data), desc='Validation'):
            batch_size = tf.shape(images)[0]
            patches = tf.reshape(images, (-1, self.target_size[0], self.target_size[1], 3))
            patch_predictions = self.model.predict(patches, batch_size=batch_size, verbose=0)
            patch_predictions = tf.reshape(patch_predictions, (batch_size, 5))
            patch_predictions = tf.reduce_mean(patch_predictions, axis=1)
            y_pred.append(patch_predictions.numpy())
            y_true.append(labels.numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        logs['val_loss'] = self.loss(y_true, y_pred)

        PLCC, SRCC, MAE, RMSE = evaluate_metrics(y_pred, y_true)
        logs['val_plcc'] = PLCC
        logs['val_srcc'] = SRCC
        logs['val_mae'] = MAE
        logs['val_rmse'] = RMSE

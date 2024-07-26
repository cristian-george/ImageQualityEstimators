import numpy as np
import tensorflow as tf
from tqdm import tqdm

from util.metrics import plcc


class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, data, loss_fn, target_size):
        super().__init__()
        self.data = data
        self.loss_fn = loss_fn
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

        loss = self.loss_fn(y_true, y_pred)
        logs['val_loss'] = loss

        mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
        logs['val_mae'] = mae

        plcc_tf = np.round(plcc(y_pred, y_true), 3)
        logs['val_plcc_tf'] = plcc_tf

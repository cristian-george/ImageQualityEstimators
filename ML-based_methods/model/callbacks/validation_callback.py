import numpy as np
import tensorflow as tf
from tqdm import tqdm

from util.metrics import plcc


class ValidationCallback(tf.keras.callbacks.Callback):
    def __init__(self, data, image_shape, input_shape):
        super().__init__()
        self.data = data
        self.image_shape = image_shape
        self.input_shape = input_shape

    def on_epoch_end(self, epoch, logs=None):
        y_pred = []
        y_true = []

        for images, labels in tqdm(self.data, total=len(self.data), desc='Validation'):
            batch_size = tf.shape(images)[0]

            if self.image_shape == self.input_shape:
                predictions = self.model.predict(images, batch_size=batch_size, verbose=0)
                predictions = tf.squeeze(predictions, axis=1)
            else:
                patches = tf.reshape(images, (-1,) + self.input_shape)
                patch_predictions = self.model.predict(patches, batch_size=batch_size, verbose=0)
                patch_predictions = tf.reshape(patch_predictions, (batch_size, 5))
                predictions = tf.reduce_mean(patch_predictions, axis=1)

            y_pred.append(predictions.numpy())
            y_true.append(labels.numpy())

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        logs['val_loss'] = self.model.loss(y_true, y_pred)
        logs['val_plcc'] = np.round(plcc(y_true, y_pred), 4)

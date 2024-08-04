from keras import Model, Sequential
from keras.layers import GlobalAveragePooling2D, Dense

from model.network.layers.dense_block import DenseBlock


class QualityPredictor(Model):
    def __init__(self, dense, dropout):
        super(QualityPredictor, self).__init__()

        self.model = Sequential()
        self.model.add(GlobalAveragePooling2D())

        for units, dropout_rate in zip(dense, dropout):
            self.model.add(DenseBlock(units, activation='relu', dropout_rate=dropout_rate))

        self.model.add(Dense(dense[-1], activation='linear', kernel_initializer='he_normal'))

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)

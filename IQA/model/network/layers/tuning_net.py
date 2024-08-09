from keras import Model, Sequential
from keras.layers import Layer, GlobalAveragePooling2D, Dense, BatchNormalization, Dropout


class TuningNet(Model):
    def __init__(self, dense, dropout):
        super(TuningNet, self).__init__()

        self.model = Sequential()
        self.model.add(GlobalAveragePooling2D())

        for units, dropout_rate in zip(dense, dropout):
            self.model.add(DenseBlock(units, activation='relu', dropout_rate=dropout_rate))

        self.model.add(Dense(dense[-1], activation='linear', kernel_initializer='he_normal'))

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)


class DenseBlock(Layer):
    """
        DenseBlock is a custom Keras layer that combines a fully connected (Dense) layer
        with Batch Normalization and Dropout.

        Parameters
        ----------
        units : int
            The number of neurons in the Dense layer.
        activation : str, optional
            The activation function to use in the Dense layer (default is 'relu').
        dropout_rate : float, optional
            The rate of dropout (default is 0.5). This value should be between 0 and 1,
            representing the fraction of input units to drop.

        Methods
        -------
        call(inputs, training=None, mask=None)
            Defines the computation from inputs to outputs.
        """

    def __init__(self, units, activation='relu', dropout_rate=0.5):
        super().__init__()

        self.dense = Dense(units, activation=activation, kernel_initializer='he_normal')
        self.batch_norm = BatchNormalization()
        self.dropout = Dropout(dropout_rate)

    def call(self, inputs, training=None, mask=None):
        x = self.dense(inputs)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        return x

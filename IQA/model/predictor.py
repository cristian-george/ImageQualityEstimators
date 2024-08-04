from keras.optimizers import Adam

from model.network.image_quality_predictor import ImageQualityPredictor
from util.metrics_tf import plcc, srcc, mae, rmse


class Predictor:
    def __init__(self, model_info):
        name = model_info.get('name', '')
        freeze = model_info.get('freeze')
        dense = model_info.get('dense', [])
        dropout = model_info.get('dropout', [])

        self.input_shape = tuple(model_info.get('input_shape', []))
        self.model = ImageQualityPredictor(name=name,
                                           freeze=freeze,
                                           input_shape=(self.input_shape[0], self.input_shape[1], 3),
                                           dense=dense,
                                           dropout=dropout)

    def summary(self):
        self.model.summary(show_trainable=True)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def compile(self, loss=None, learning_rate=0.001):
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss=loss,
                           metrics=[plcc, srcc, mae, rmse])

    def fit(self, data, epochs, initial_epoch, validation_data, callbacks):
        return self.model.fit(data,
                              epochs=epochs,
                              initial_epoch=initial_epoch,
                              validation_data=validation_data,
                              callbacks=callbacks)

    def predict(self, data, batch_size):
        prediction = self.model.predict(data, batch_size=batch_size, verbose=0)
        return prediction

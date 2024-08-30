from keras import Model
from keras.applications import ResNet50, VGG16
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout
from keras.optimizers import Adam

from config_parser.model_config_parser import ModelConfigParser
from util.metrics_tf import plcc, srcc, mae, rmse


class Predictor:
    def __init__(self):
        self.config_parser = ModelConfigParser()
        self.model_info = self.config_parser.parse()

        self.__init_model_info()
        self.__build_model()

    def __init_model_info(self):
        self.backbone = self.model_info['backbone']
        self.freeze = self.model_info['freeze']
        self.input_shape = self.model_info['input_shape']
        self.dense = self.model_info['dense']
        self.dropout = self.model_info['dropout']

    def __get_backbone_net(self):
        if self.backbone == 'resnet50':
            backbone_net = ResNet50(weights='imagenet',
                                    include_top=False,
                                    input_shape=self.input_shape)
        elif self.backbone == 'vgg16':
            backbone_net = VGG16(weights='imagenet',
                                 include_top=False,
                                 input_shape=self.input_shape)
        else:
            raise ValueError('Invalid backbone model')

        if self.freeze:
            for layer in backbone_net.layers:
                layer.trainable = False

        return backbone_net.input, backbone_net.layers[-2].output

    def __get_tuning_net(self, inputs):
        x = GlobalAveragePooling2D()(inputs)
        for units, rate in zip(self.dense, self.dropout):
            x = Dense(units,
                      activation='relu',
                      kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Dropout(rate)(x)

        return Dense(self.dense[-1],
                     activation='linear',
                     kernel_initializer='he_normal')(x)

    def __build_model(self):
        # Backbone net
        backbone_input, backbone_output = self.__get_backbone_net()

        # Tuning net
        tuning_output = self.__get_tuning_net(backbone_output)

        self.model = Model(name='image_quality_predictor',
                           inputs=backbone_input,
                           outputs=tuning_output)

    def summary(self):
        self.model.summary(show_trainable=True)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def compile(self, loss=None, learning_rate=0.001):
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss=loss,
                           metrics=[plcc, srcc, mae, rmse])

    def fit(self, data, epochs, initial_epoch, callbacks, validation_data=None):
        return self.model.fit(data,
                              validation_data=validation_data,
                              epochs=epochs,
                              initial_epoch=initial_epoch,
                              callbacks=callbacks)

    def predict(self, data, batch_size):
        prediction = self.model.predict(data, batch_size=batch_size, verbose=0)
        return prediction

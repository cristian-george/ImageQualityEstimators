import keras
from keras import Model
from keras.applications import ResNet50, VGG16
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Flatten, Dense, BatchNormalization, \
    Dropout
from keras.optimizers import Adam

from config_parser.model_config_parser import ModelConfigParser
from model.vars import network_architecture
from util.metrics_tf import plcc, srcc, mae, rmse

networks = {
    'vgg16': VGG16,
    'resnet50': ResNet50,
}

output_layer = {
    'vgg16': -1,
    'resnet50': -2,
}

preprocessing_functions = {
    'vgg16': keras.applications.vgg16.preprocess_input,
    'resnet50': keras.applications.resnet50.preprocess_input,
}


class Predictor:
    def __init__(self):
        self.config_parser = ModelConfigParser()
        self.model_info = self.config_parser.parse()

        self.__init_model_info()
        self.__build_model()

    def __init_model_info(self):
        self.net_name = self.model_info['net_name']
        self.input_shape = self.model_info['input_shape']
        self.freeze = self.model_info['freeze']
        self.pooling = self.model_info['pooling']
        self.dense = self.model_info['dense']
        self.dropout = self.model_info['dropout']

    def __get_backbone_net(self):
        if self.net_name not in list(network_architecture.keys()):
            raise ValueError('Invalid network architecture')

        network: [VGG16 | ResNet50] = network_architecture[self.net_name]
        backbone: [VGG16 | ResNet50] = network(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape)

        if self.freeze:
            for layer in backbone.layers:
                layer.trainable = False

        if self.net_name == 'resnet50':
            feats = backbone.layers[-2]
        else:
            feats = backbone.layers[-1]

        return backbone.input, feats.output

    def __get_tuning_net(self, inputs):
        if self.pooling == 'max':
            x = MaxPooling2D((2, 2), strides=(2, 2))(inputs)
            x = Flatten()(x)
        elif self.pooling == 'avg':
            x = AveragePooling2D((2, 2), strides=(2, 2))(inputs)
            x = Flatten()(x)
        elif self.pooling == 'global_avg':
            x = GlobalAveragePooling2D()(inputs)
        else:
            x = Flatten()(inputs)

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

from keras import Model
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, \
    BatchNormalization, Dropout, Activation
from keras.optimizers import Adam

from config_parser.model_config_parser import ModelConfigParser
from model.vars import network_architecture
from util.tf_metrics import plcc


class Predictor:
    def __init__(self):
        config_parser = ModelConfigParser()
        self.config = config_parser.parse()

        self.__init_fields()
        self.__build_model()

    def __init_fields(self):
        self.net_name = self.config['net_name']
        self.input_shape = self.config['input_shape']
        self.freeze_backbone = self.config['freeze_backbone']
        self.freeze_head_bn = self.config['freeze_head_bn']
        self.pooling = self.config['pooling']
        self.dense = self.config['dense']
        self.dropout = self.config['dropout']

    def __build_model(self):
        # Backbone net
        backbone_input, backbone_output = self.__get_backbone_net()

        # Tuning net
        tuning_output = self.__get_tuning_net(backbone_output)

        self.model = Model(name='image_quality_predictor',
                           inputs=backbone_input,
                           outputs=tuning_output)

    def __get_backbone_net(self):
        if self.net_name not in list(network_architecture.keys()):
            raise ValueError('Invalid network architecture')

        network = network_architecture[self.net_name]
        backbone = network(weights='imagenet',
                           include_top=False,
                           input_shape=self.input_shape)

        backbone.trainable = False
        for layer in backbone.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = not self.freeze_backbone

        if self.net_name == 'vgg16':
            feats = backbone.layers[-2]
        else:
            feats = backbone.layers[-1]

        return backbone.input, feats.output

    def __get_tuning_net(self, inputs):
        if self.pooling == 'avg':
            x = AveragePooling2D((2, 2), strides=(2, 2))(inputs)
            x = Flatten()(x)
        elif self.pooling == 'max':
            x = MaxPooling2D((2, 2), strides=(2, 2))(inputs)
            x = Flatten()(x)
        elif self.pooling == 'global_avg':
            x = GlobalAveragePooling2D()(inputs)
        elif self.pooling == 'global_max':
            x = GlobalMaxPooling2D()(inputs)
        else:
            raise ValueError('Invalid pooling type')

        for units, rate in zip(self.dense, self.dropout):
            x = Dense(units,
                      kernel_initializer='he_normal')(x)

            if self.net_name != 'vgg16':
                bn = BatchNormalization()
                bn.trainable = not self.freeze_head_bn
                x = bn(x)

            x = Activation('relu')(x)
            x = Dropout(rate)(x)

        return Dense(self.dense[-1],
                     kernel_initializer='he_normal')(x)

    def summary(self):
        self.model.summary(show_trainable=True)
        print(f"No. of layers: {len(self.model.layers)}")

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def compile(self, loss=None, learning_rate=0.001):
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss=loss,
                           metrics=[plcc])

    def fit(self, data, epochs, initial_epoch, callbacks, validation_data=None):
        return self.model.fit(data,
                              validation_data=validation_data,
                              epochs=epochs,
                              initial_epoch=initial_epoch,
                              callbacks=callbacks)

    def predict(self, data, batch_size):
        prediction = self.model.predict(data, batch_size=batch_size, verbose=0)
        return prediction

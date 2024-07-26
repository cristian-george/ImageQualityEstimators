from keras.applications import ResNet50
from keras.applications import VGG16
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

from util.metrics import plcc_tf


class ImageQualityPredictor(object):
    def __init__(self, model_info):
        self.model_info = model_info

        self.__init_model_info()
        self.__build_model()

    def __init_model_info(self):
        self.name = self.model_info.get('name', '')
        self.freeze = self.model_info.get('freeze')
        self.input_shape = tuple(self.model_info.get('input_shape', []))
        self.dense = self.model_info.get('dense', [])
        self.dropout = self.model_info.get('dropout', [])

    def __build_model(self):
        base_model = None
        match self.name:
            case 'resnet50':
                base_model = ResNet50(weights='imagenet',
                                      include_top=False,
                                      input_shape=(self.input_shape[0], self.input_shape[1], 3))

            case 'vgg16':
                base_model = VGG16(weights='imagenet',
                                   include_top=False,
                                   input_shape=(self.input_shape[0], self.input_shape[1], 3))

        if self.freeze:
            for layer in base_model.layers[:-2]:
                layer.trainable = False

        features = base_model.layers[-2].output
        x = GlobalAveragePooling2D()(features)
        for i in range(3):
            x = Dense(self.dense[i],
                      activation='relu',
                      kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout[i])(x)

        mos_output = Dense(self.dense[3],
                           activation='linear',
                           kernel_initializer='he_normal')(x)

        self.model = Model(inputs=base_model.input,
                           outputs=mos_output)

    def summary(self):
        self.model.summary(show_trainable=True)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def compile(self, loss, learning_rate=0.001):
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss=loss,
                           metrics=['mae', plcc_tf])

    def fit(self, data, epochs, initial_epoch, validation_data, callbacks):
        return self.model.fit(data,
                              epochs=epochs,
                              initial_epoch=initial_epoch,
                              validation_data=validation_data,
                              callbacks=callbacks)

    def predict(self, data, batch_size):
        prediction = self.model.predict(data, batch_size=batch_size, verbose=0)
        return prediction

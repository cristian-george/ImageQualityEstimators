from keras import Input
from keras.models import Model

from model.network.layers.feature_extractor import FeatureExtractor
from model.network.layers.quality_predictor import QualityPredictor


class ImageQualityPredictor(Model):
    def __init__(self, name='resnet50', input_shape=(224, 224, 3), freeze=True, dense=None, dropout=None):
        super(ImageQualityPredictor, self).__init__()

        self.input_layer = Input(name='input_layer',
                                 shape=input_shape)

        self.feature_extractor = FeatureExtractor(name=name,
                                                  input_shape=input_shape,
                                                  freeze=freeze)

        self.quality_predictor = QualityPredictor(dense=dense,
                                                  dropout=dropout)

        self.output_layer = self.call(self.input_layer)

        super(ImageQualityPredictor, self).__init__(inputs=self.input_layer,
                                                    outputs=self.output_layer)

    def call(self, inputs, training=None, mask=None):
        features = self.feature_extractor(inputs)
        score = self.quality_predictor(features, training=training)
        return score

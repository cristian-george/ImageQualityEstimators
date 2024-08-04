from keras import Model
from keras.applications import ResNet50, VGG16


class FeatureExtractor(Model):
    def __init__(self, name='resnet50', input_shape=(224, 224, 3), freeze=True):
        super(FeatureExtractor, self).__init__()

        if name == 'resnet50':
            backbone = ResNet50(weights='imagenet',
                                include_top=False,
                                input_shape=input_shape)
        elif name == 'vgg16':
            backbone = VGG16(weights='imagenet',
                             include_top=False,
                             input_shape=input_shape)
        else:
            raise ValueError('Invalid base model')

        self.model = Model(inputs=backbone.input,
                           outputs=backbone.layers[-2].output)

        if freeze:
            self.trainable = False
            for layer in self.model.layers:
                layer.trainable = False

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)

    @property
    def output(self):
        return self.model.output

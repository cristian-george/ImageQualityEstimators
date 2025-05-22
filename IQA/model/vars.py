import keras
from keras.applications import VGG16, ResNet50, InceptionV3, NASNetMobile, EfficientNetV2S

network_architecture = {
    'vgg16': VGG16,
    'resnet50': ResNet50,
    'inception_v3': InceptionV3,
    'nasnet_mobile': NASNetMobile,
    'efficientnet_v2_s': EfficientNetV2S,
}

preprocess_function = {
    'vgg16': keras.applications.vgg16.preprocess_input,
    'resnet50': keras.applications.resnet.preprocess_input,
    'inception_v3': keras.applications.inception_v3.preprocess_input,
    'nasnet_mobile': keras.applications.nasnet.preprocess_input,
    'efficientnet_v2_s': keras.applications.efficientnet_v2.preprocess_input,
}

import keras
from keras.applications import VGG16, ResNet50

network_architecture = {
    'resnet50': ResNet50,
    'vgg16': VGG16,
}

preprocess_function = {
    'resnet50': keras.applications.resnet50.preprocess_input,
    'vgg16': keras.applications.vgg16.preprocess_input,
}

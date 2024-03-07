import os

import tensorflow as tf
import keras.callbacks
import pandas as pd
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from models.metrics import srcc, plcc

from util.gpu_utils import check_gpu_support, limit_gpu_memory, increase_cpu_num_threads

image_size = (512, 384)
batch_size: int = 10
epochs: int = 40


def build_model():
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(image_size[0], image_size[1], 3))

    # for layer in base_model.layers:
    #     layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='linear')(x)
    return Model(inputs=base_model.input, outputs=output)


def compile_model(model):
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss=tf.keras.losses.Huber(),
                  metrics=[srcc, plcc])


def train_model(model, train_dir, val_dir, train_labels_file, val_labels_file):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_df = pd.read_csv(train_labels_file)
    val_df = pd.read_csv(val_labels_file)

    train_generator = train_datagen.flow_from_dataframe(
        train_df, directory=train_dir, x_col='image_name', y_col='MOS',
        target_size=image_size, batch_size=batch_size, class_mode='raw')

    validation_generator = val_datagen.flow_from_dataframe(
        val_df, directory=val_dir, x_col='image_name', y_col='MOS',
        target_size=image_size, batch_size=batch_size, class_mode='raw', shuffle=False)

    tensorboard_callback = TensorBoard(log_dir='../../logs', histogram_freq=1)
    early_stopping_callback = keras.callbacks.EarlyStopping(patience=5,
                                                            monitor='loss')

    ckpt_dir = '../../trained_models/vgg16_finetune_huber'
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    best_loss_callback = ModelCheckpoint(os.path.join(ckpt_dir, 'model_{epoch:02d}.h5'),
                                         monitor='val_loss',
                                         save_best_only=True,
                                         save_weights_only=True)

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=[tensorboard_callback, early_stopping_callback, best_loss_callback])


def evaluate_model(model, test_dir, test_labels_file):
    test_df = pd.read_csv(test_labels_file)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_dataframe(
        test_df, directory=test_dir, x_col='image_name', y_col='MOS',
        target_size=image_size, batch_size=batch_size, class_mode='raw')

    val_loss = model.evaluate(
        test_generator,
        steps=test_generator.samples // batch_size)
    print(f'Values (huber, srcc, plcc): {val_loss}')


if __name__ == "__main__":
    use_gpu = check_gpu_support()
    if use_gpu:
        limit_gpu_memory(memory_limit=3600)
    else:
        increase_cpu_num_threads(num_threads=os.cpu_count())

    data_directory = '../../data/'
    train_directory = data_directory + 'train/all_classes'
    val_directory = data_directory + 'validation/all_classes'
    train_lb = data_directory + 'train_labels.csv'
    val_lb = data_directory + 'val_labels.csv'

    model = build_model()
    # model.load_weights('../trained_models/best_checkpoint_resnet50/model_38.h5')
    compile_model(model)

    model.summary()

    train_model(model, train_directory, val_directory, train_lb, val_lb)

    test_directory = data_directory + 'test/all_classes'
    test_lb = data_directory + 'test_labels.csv'

    evaluate_model(model, test_directory, test_lb)

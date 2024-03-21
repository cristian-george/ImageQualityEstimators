import os
import pandas as pd
import tensorflow as tf
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing import image
from keras.applications.resnet import preprocess_input

from model.metrics import plcc_tf, rmse_tf, srcc_tf
from model.scheduler import LRSchedule
from util.random_crop import crop_generator


class IQA:
    def __init__(self, model_info, learn_info, train_info, evaluate_info):
        self.model_info = model_info
        self.learn_info = learn_info
        self.train_info = train_info
        self.evaluate_info = evaluate_info

        self.__init_model_info()
        self.__build_model()

    def __init_model_info(self):
        self.model_name = self.model_info.get('name', '')
        self.image_original_size = tuple(self.model_info.get('image_original_size', []))
        self.model_input_shape = tuple(self.model_info.get('model_input_shape', []))

        self.batch_size = self.model_info.get('batch_size')
        self.epoch_size = self.model_info.get('epoch_size')
        self.crop_image = self.model_info.get('crop_image')

    def __build_model(self):
        base_model = None
        match self.model_name:
            case 'resnet50':
                base_model = ResNet50(weights='imagenet',
                                      include_top=False,
                                      input_shape=(self.model_input_shape[0], self.model_input_shape[1], 3))

            case 'vgg16':
                base_model = VGG16(weights='imagenet',
                                   include_top=False,
                                   input_shape=(self.model_input_shape[0], self.model_input_shape[1], 3))

        freeze = self.model_info.get('freeze')
        if freeze:
            for layer in base_model.layers[-2]:
                layer.trainable = False

        dense = self.model_info.get('dense', [])
        dropout = self.model_info.get('dropout', [])

        features = base_model.layers[-2].output
        x = GlobalAveragePooling2D()(features)
        for i in range(3):
            x = Dense(dense[i],
                      activation='relu',
                      kernel_initializer='he_normal')(x)
            # x = BatchNormalization()(x)
            x = Dropout(dropout[i])(x)

        mos_output = Dense(dense[3],
                           activation='linear',
                           kernel_initializer='he_normal')(x)

        self.model = Model(inputs=base_model.input,
                           outputs=mos_output)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def compile_model(self, total_batches=0):
        lr = self.learn_info.get('lr')
        if lr:
            learning_rate = lr
        else:
            custom_scheduler = self.learn_info.get('custom_scheduler', {})
            decay_epochs = custom_scheduler.get('epoch_decays', [])
            learning_rates = custom_scheduler.get('learning_rates', [])

            learning_rate = LRSchedule(decay_epochs=decay_epochs,
                                       learning_rates=learning_rates,
                                       total_batches=total_batches,
                                       total_epochs=self.epoch_size)

        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss='mse',
                           metrics=['mae', rmse_tf, plcc_tf, srcc_tf])

    def __callbacks(self):
        callbacks_info = self.train_info.get('callbacks', {})
        tensorboard_info = callbacks_info.get('tensorboard_checkpoint', {})
        log_dir = tensorboard_info.get('log_dir', '')
        histogram_freq = tensorboard_info.get('histogram_freq')

        tensorboard_callback = TensorBoard(log_dir=log_dir,
                                           histogram_freq=histogram_freq)

        best_model_checkpoint = callbacks_info.get('best_checkpoint', {})
        ckpt_dir = best_model_checkpoint.get('ckpt_dir', '')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        monitor = best_model_checkpoint.get('monitor', '')
        mode = best_model_checkpoint.get('mode', '')
        save_best_only = best_model_checkpoint.get('save_best_only')
        save_weights_only = best_model_checkpoint.get('save_weights_only')

        best_model_callback = ModelCheckpoint(os.path.join(ckpt_dir, 'best_model.h5'),
                                              monitor=monitor,
                                              mode=mode,
                                              save_best_only=save_best_only,
                                              save_weights_only=save_weights_only)

        early_stopping_callback = EarlyStopping(monitor=monitor,
                                                mode=mode,
                                                patience=20)

        return [tensorboard_callback, best_model_callback, early_stopping_callback]

    def train_model(self):
        data_directory = self.train_info.get('data_directory', '')
        train_dir = data_directory + self.train_info.get('train_directory', '')
        val_dir = data_directory + self.train_info.get('val_directory', '')
        train_labels_file = data_directory + self.train_info.get('train_lb', '')
        val_labels_file = data_directory + self.train_info.get('val_lb', '')

        augment = self.train_info.get('augment')
        if augment:
            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                               horizontal_flip=True)
        else:
            train_datagen = ImageDataGenerator(rescale=1. / 255)

        val_datagen = ImageDataGenerator(rescale=1. / 255)

        train_df = pd.read_csv(train_labels_file)
        val_df = pd.read_csv(val_labels_file)

        target_size = self.image_original_size if self.crop_image else self.model_input_shape
        train_generator = train_datagen.flow_from_dataframe(
            train_df, directory=train_dir, x_col='image_name', y_col='MOS',
            target_size=target_size, interpolation='lanczos',
            batch_size=self.batch_size, class_mode='raw')

        valid_generator = val_datagen.flow_from_dataframe(
            val_df, directory=val_dir, x_col='image_name', y_col='MOS',
            target_size=target_size, interpolation='lanczos',
            batch_size=self.batch_size, class_mode='raw')

        crop_train = None
        crop_valid = None
        if self.crop_image:
            crop_train = crop_generator(train_generator, self.model_input_shape)
            crop_valid = crop_generator(valid_generator, self.model_input_shape)

        weights_path = self.train_info.get('continue_train', '')
        initial_epoch = self.train_info.get('initial_epoch')

        if weights_path and initial_epoch:
            self.load_weights(weights_path)

        steps_per_epoch = train_generator.samples // self.batch_size
        self.compile_model(steps_per_epoch)

        self.model.fit(
            crop_train if self.crop_image else train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epoch_size + initial_epoch,
            initial_epoch=initial_epoch,
            validation_data=crop_valid if self.crop_image else valid_generator,
            validation_steps=valid_generator.samples // self.batch_size,
            callbacks=self.__callbacks())

    def evaluate_model(self):
        data_directory = self.evaluate_info.get('data_directory', '')
        test_dir = data_directory + self.evaluate_info.get('test_directory', '')
        test_labels_file = data_directory + self.evaluate_info.get('test_lb', '')

        weights_path = self.evaluate_info.get('weights_path', '')

        self.load_weights(weights_path)
        self.compile_model()

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        test_df = pd.read_csv(test_labels_file)

        target_size = self.image_original_size if self.crop_image else self.model_input_shape
        test_generator = test_datagen.flow_from_dataframe(
            test_df, directory=test_dir, x_col='image_name', y_col='MOS',
            target_size=target_size, interpolation='lanczos',
            batch_size=self.batch_size, class_mode='raw')

        crop_test = None
        if self.crop_image:
            crop_test = crop_generator(test_generator, self.model_input_shape)

        val_loss = self.model.evaluate(
            crop_test if self.crop_image else test_generator,
            steps=test_generator.samples // self.batch_size)
        print(f'Values (mse, mae, rmse_tf, plcc_tf, srcc_tf): {val_loss}')

    def predict_score_for_image(self, image_path):
        img = image.load_img(image_path, target_size=self.model_input_shape)

        img_array = image.img_to_array(img)
        img_array /= 255.0

        img_tensor = tf.expand_dims(img_array, axis=0)

        score = self.model.predict(img_tensor, verbose=0)

        return score[0][0]

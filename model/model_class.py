import os

import keras.losses
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from keras.applications import ResNet50
from keras.applications import VGG16
from keras.applications.resnet import preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing import image

from model.metrics import plcc_tf
from model.staircase_schedule import StaircaseSchedule
from util.random_crop import crop_generator
from util.resize_crop import resize_and_crop


class IQA:
    def __init__(self, model_info, train_info, evaluate_info):
        self.model = None
        self.model_info = model_info
        self.train_info = train_info
        self.evaluate_info = evaluate_info

        self.__init_model_info()
        self.__init_train_info()
        self.__init_evaluate_info()

    def __init_model_info(self):
        self.name = self.model_info.get('name', '')
        self.freeze = self.model_info.get('freeze')
        self.input_shape = tuple(self.model_info.get('input_shape', []))
        self.dense = self.model_info.get('dense', [])
        self.dropout = self.model_info.get('dropout', [])

    def __init_train_info(self):
        self.data_directory = self.train_info.get('data_directory', '')
        self.train_directory = self.data_directory + self.train_info.get('train_directory', '')
        self.val_directory = self.data_directory + self.train_info.get('val_directory', '')
        self.train_lb = self.data_directory + self.train_info.get('train_lb', '')
        self.val_lb = self.data_directory + self.train_info.get('val_lb', '')

        self.image_size = tuple(self.train_info.get('image_size', []))
        self.crop_image = self.train_info.get('crop_image')
        self.augment = self.train_info.get('augment')

        self.batch_size = self.train_info.get('batch_size')
        self.epoch_size = self.train_info.get('epoch_size')

        self.continue_train = self.train_info.get('continue_train', {})

    def __init_evaluate_info(self):
        self.root_directory = self.evaluate_info.get('root_directory', '')
        self.test_directory = self.root_directory + self.evaluate_info.get('test_directory', '')
        self.test_lb = self.root_directory + self.evaluate_info.get('test_lb', '')
        self.weights_path = self.evaluate_info.get('weights_path', '')

    def build_model(self):
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

    def compile(self, total_batches=0):
        learning_rate = self.__get_learning_rate(total_batches)
        loss = self.__get_loss()

        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss=loss,
                           metrics=['mae', plcc_tf])

    def __get_learning_rate(self, total_batches):
        lr = self.train_info.get('lr', {})
        name = lr.get('name', '')

        match name:
            case "constant":
                return lr.get('value')

            case "exponential_decay":
                exponential_decay = lr.get('value', {})
                initial_learning_rate = exponential_decay.get('initial_learning_rate')
                final_learning_rate = exponential_decay.get('final_learning_rate')
                learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / self.epoch_size)
                staircase = exponential_decay.get('staircase')

                return ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                        decay_steps=total_batches,
                                        decay_rate=learning_rate_decay_factor,
                                        staircase=staircase)

            case "staircase_scheduler":
                staircase_scheduler = lr.get('value', {})
                decay_epochs = staircase_scheduler.get('epoch_decays', [])
                learning_rates = staircase_scheduler.get('learning_rates', [])

                return StaircaseSchedule(decay_epochs=decay_epochs,
                                         learning_rates=learning_rates,
                                         total_batches=total_batches,
                                         total_epochs=self.epoch_size)

    def __get_loss(self):
        loss = self.train_info.get('loss', {})
        name = loss.get('name')

        match name:
            case 'huber':
                delta = loss.get('delta')
                return keras.losses.Huber(delta=delta)
            case 'mse':
                return name

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
        if self.augment:
            train_datagen = ImageDataGenerator(horizontal_flip=True,
                                               preprocessing_function=preprocess_input)
        else:
            train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        train_df = pd.read_csv(self.train_lb)
        val_df = pd.read_csv(self.val_lb)

        target_size = self.image_size if self.crop_image else self.input_shape
        train_generator = train_datagen.flow_from_dataframe(
            train_df, directory=self.train_directory, x_col='image_name', y_col='MOS',
            target_size=target_size, interpolation='lanczos',
            batch_size=self.batch_size, class_mode='raw')

        valid_generator = val_datagen.flow_from_dataframe(
            val_df, directory=self.val_directory, x_col='image_name', y_col='MOS',
            target_size=target_size, interpolation='lanczos',
            batch_size=self.batch_size, class_mode='raw')

        crop_train = None
        crop_valid = None
        if self.crop_image:
            crop_train = crop_generator(train_generator, self.input_shape)
            crop_valid = crop_generator(valid_generator, self.input_shape)

        continue_train_from_epoch = self.continue_train.get('from_epoch')
        continue_train_from_weights = self.continue_train.get('from_weights', '')

        if continue_train_from_epoch and continue_train_from_weights:
            self.load_weights(continue_train_from_weights)

        steps_per_epoch = train_generator.samples // self.batch_size
        self.compile(steps_per_epoch)

        self.model.fit(
            crop_train if self.crop_image else train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epoch_size + continue_train_from_epoch,
            initial_epoch=continue_train_from_epoch,
            validation_data=crop_valid if self.crop_image else valid_generator,
            validation_steps=valid_generator.samples // self.batch_size,
            callbacks=self.__callbacks())

    def evaluate_model(self):
        self.load_weights(self.weights_path)
        self.compile()

        test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

        test_df = pd.read_csv(self.test_lb)

        target_size = self.image_size if self.crop_image else self.input_shape
        test_generator = test_datagen.flow_from_dataframe(
            test_df, directory=self.test_directory, x_col='image_name', y_col='MOS',
            target_size=target_size, interpolation='lanczos',
            batch_size=self.batch_size, class_mode='raw')

        crop_test = None
        if self.crop_image:
            crop_test = crop_generator(test_generator, self.input_shape)

        val_loss = self.model.evaluate(
            crop_test if self.crop_image else test_generator,
            steps=test_generator.samples // self.batch_size)
        print(f'Values (loss, mae, plcc_tf): {val_loss}')

    def predict_score_for_image(self, image_path, keep_aspect_ratio):
        if keep_aspect_ratio:
            img = image.load_img(image_path)
            img_array = image.img_to_array(img)
            crops = resize_and_crop(img_array, self.input_shape)
            scores = []
            for crop in crops:
                crop = preprocess_input(crop)
                crop_tensor = tf.expand_dims(crop, axis=0)
                score = self.model.predict(crop_tensor, verbose=0)
                scores.append(score[0][0])

            return np.median(scores)

        img = image.load_img(image_path,
                             target_size=self.input_shape,
                             interpolation="lanczos")
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_tensor = tf.expand_dims(img_array, axis=0)
        score = self.model.predict(img_tensor, verbose=0)

        return score[0][0]

    def predict_scores(self, image_names, keep_aspect_ratio=False):
        self.load_weights(self.weights_path)
        self.compile()

        predicted_scores = []
        for image_name in tqdm.tqdm(image_names, desc="Predict scores"):
            image_path = self.test_directory + "/" + image_name

            # Predict score for the image
            predicted_score = self.predict_score_for_image(image_path, keep_aspect_ratio)
            predicted_scores.append(predicted_score)
        return predicted_scores

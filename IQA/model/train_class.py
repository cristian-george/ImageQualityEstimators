import os

import keras.losses
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay

from model.model_class import ImageQualityPredictor
from util.preprocess_input import crop_5patches, static_preprocess_image, dynamic_preprocess_image


class PredictorTrainer:
    def __init__(self, train_info, model: ImageQualityPredictor):
        self.train_info = train_info
        self.model = model

        self.__init_train_info()

    def __init_train_info(self):
        self.data_directory = self.train_info.get('data_directory', '')
        self.train_directory = self.data_directory + self.train_info.get('train_directory', '')
        self.val_directory = self.data_directory + self.train_info.get('val_directory', '')
        self.train_lb = self.data_directory + self.train_info.get('train_lb', '')
        self.val_lb = self.data_directory + self.train_info.get('val_lb', '')

        self.augment = self.train_info.get('augment')

        self.batch_size = self.train_info.get('batch_size')
        self.epoch_size = self.train_info.get('epoch_size')

        self.continue_train = self.train_info.get('continue_train', {})

    def __get_learning_rate(self, steps_per_epoch):
        lr = self.train_info.get('lr', {})
        self.name = lr.get('name', '')

        match self.name:
            case "constant":
                return lr.get('value')

            case "exponential_decay":
                exponential_decay = lr.get('value', {})
                initial_learning_rate = exponential_decay.get('initial_learning_rate')
                final_learning_rate = exponential_decay.get('final_learning_rate')
                learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / self.epoch_size)
                staircase = exponential_decay.get('staircase')

                return ExponentialDecay(initial_learning_rate=initial_learning_rate,
                                        decay_steps=steps_per_epoch,
                                        decay_rate=learning_rate_decay_factor,
                                        staircase=staircase)

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

    def static_preprocess_img(self, x, y):
        x = static_preprocess_image(x)
        return x, y

    def dynamic_preprocess_img(self, x, y):
        x = dynamic_preprocess_image(x, self.model.input_shape, self.augment)
        return x, y

    def __create_train_set_from_dataframe(self, train_df):
        image_paths = train_df['image_name'].apply(lambda image_path: self.train_directory + "/" + image_path)
        labels = train_df['MOS']

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(self.static_preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
        dataset.cache()

        dataset = dataset.map(self.dynamic_preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def __create_validation_set_from_dataframe(self, val_df):
        image_paths = val_df['image_name'].apply(lambda image_path: self.val_directory + "/" + image_path)
        labels = val_df['MOS']

        # Shuffle validation data once
        indices = np.arange(len(image_paths))
        np.random.shuffle(indices)
        image_paths = np.array(image_paths)[indices]
        labels = np.array(labels)[indices]

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(self.static_preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.dynamic_preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset.cache()

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def fit_model(self):
        train_df = pd.read_csv(self.train_lb)
        validation_df = pd.read_csv(self.val_lb)

        train_dataset = self.__create_train_set_from_dataframe(train_df)
        validation_dataset = self.__create_validation_set_from_dataframe(validation_df)

        continue_train_from_epoch = self.continue_train.get('from_epoch')
        continue_train_from_weights = self.continue_train.get('from_weights', '')

        if continue_train_from_epoch and continue_train_from_weights:
            self.model.load_weights(continue_train_from_weights)

        steps_per_epoch = train_df.shape[0] // self.batch_size
        validation_steps = validation_df.shape[0] // self.batch_size

        self.model.compile(loss=self.__get_loss(),
                           learning_rate=self.__get_learning_rate(steps_per_epoch))

        return self.model.fit(train_dataset.repeat(),
                              steps_per_epoch=steps_per_epoch,
                              epochs=self.epoch_size + continue_train_from_epoch,
                              initial_epoch=continue_train_from_epoch,
                              validation_data=validation_dataset,
                              validation_steps=validation_steps,
                              callbacks=self.__callbacks())

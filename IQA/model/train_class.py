import os

import keras.losses
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay

from model.model_class import ImageQualityPredictor
from util.callbacks import ValidationCallback
from util.preprocess_input import crop_5patches, load_and_preprocess_input, crop_and_flip_input


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
                return keras.losses.MeanSquaredError()

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

    def __create_train_set_from_dataframe(self, train_df):
        image_paths = train_df['image_name'].apply(lambda image_path: self.train_directory + "/" + image_path)
        labels = train_df['MOS']

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(lambda x, y: (load_and_preprocess_input(x), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset.cache()

        dataset = dataset.map(lambda x, y: (crop_and_flip_input(x, self.model.input_shape, self.augment), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(dataset.cardinality(), reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def __create_validation_set_from_dataframe(self, val_df):
        image_paths = val_df['image_name'].apply(lambda image_path: self.val_directory + "/" + image_path)
        labels = val_df['MOS']

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(lambda x, y: (load_and_preprocess_input(x), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(dataset.cardinality(),
                                  reshuffle_each_iteration=True)

        dataset = dataset.map(lambda x, y: (crop_5patches(x, self.model.input_shape), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset.cache()

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def fit_model(self):
        continue_train_from_epoch = self.continue_train.get('from_epoch')
        continue_train_from_weights = self.continue_train.get('from_weights', '')

        if continue_train_from_epoch and continue_train_from_weights:
            self.model.load_weights(continue_train_from_weights)

        train_df = pd.read_csv(self.train_lb)
        val_df = pd.read_csv(self.val_lb)

        train_dataset = self.__create_train_set_from_dataframe(train_df)
        val_dataset = self.__create_validation_set_from_dataframe(val_df)

        loss = self.__get_loss()
        learning_rate = self.__get_learning_rate(steps_per_epoch=train_df.shape[0] // self.batch_size)
        self.model.compile(loss=loss,
                           learning_rate=learning_rate)

        validation_callback = ValidationCallback(data=val_dataset,
                                                 loss=loss,
                                                 target_size=self.model.input_shape)

        return self.model.fit(train_dataset,
                              epochs=self.epoch_size + continue_train_from_epoch,
                              initial_epoch=continue_train_from_epoch,
                              callbacks=[validation_callback] + self.__callbacks())

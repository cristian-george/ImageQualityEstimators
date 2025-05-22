import json
import os

import keras.losses
import pandas as pd

from config_parser.train_config_parser import TrainConfigParser
from model.predictor import Predictor
from model.callbacks.model_checkpoint_callbacks import get_model_checkpoint_callbacks
from model.callbacks.tensorboard_callback import get_tensorboard_callback
from model.callbacks.validation_callback import ValidationCallback
from util.tf_dataset_pipeline import create_dataset_pipeline
from model.schedulers.exponential_decay import get_exponential_decay
from model.schedulers.step_decay import get_step_decay


class PredictorTrainer:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

        self.config_parser = TrainConfigParser()
        self.train_info = self.config_parser.parse()

        self.__init_train_info()

    def __init_train_info(self):
        self.train_directory = self.train_info['train_directory']
        self.val_directory = self.train_info['val_directory']
        self.train_lb = self.train_info['train_lb']
        self.val_lb = self.train_info['val_lb']
        self.augment = self.train_info['augment']
        self.batch_size = self.train_info['batch_size']
        self.epoch_size = self.train_info['epoch_size']

    def __save_train_config(self):
        callbacks_info = self.train_info['callbacks']
        model_checkpoint_info = callbacks_info.get('model_checkpoint', {})
        ckpt_dir = model_checkpoint_info.get('ckpt_dir', '')

        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

            config = {
                "model_config": self.predictor.config.get_config_data(),
                "train_config": self.config_parser.get_config_data(),
            }
            with open(os.path.join(ckpt_dir, 'config_model.json'), 'w') as file:
                json.dump(config, file, indent=2)

    def __train_from_weights(self):
        continue_train = self.train_info['continue_train']

        self.weights = continue_train.get('from_weights', '')
        self.initial_epoch = continue_train.get('from_epoch', 0)

        if self.weights and self.initial_epoch >= 0:
            self.predictor.load_weights(self.weights)

    def __get_train_dataset(self):
        train_df = pd.read_csv(self.train_lb)

        return create_dataset_pipeline(
            train_df,
            self.train_directory,
            subset='train',
            net_name=self.predictor.net_name,
            batch_size=self.batch_size,
            target_size=self.predictor.input_shape,
            augment=self.augment)

    def __get_val_dataset(self):
        val_df = pd.read_csv(self.val_lb)

        return create_dataset_pipeline(
            val_df,
            self.val_directory,
            subset='validation',
            net_name=self.predictor.net_name,
            batch_size=self.batch_size,
            target_size=self.predictor.input_shape)

    def __get_loss(self):
        loss = self.train_info['loss']

        match loss:
            case 'mse':
                return keras.losses.MeanSquaredError()
            case 'mae':
                return keras.losses.MeanAbsoluteError()

        return None

    def __get_learning_rate(self, steps_per_epoch):
        lr = self.train_info['lr']
        name = lr.get('name', '')

        match name:
            case "constant":
                return lr.get('value', 0)

            case "exponential_decay":
                scheduler = get_exponential_decay(lr, steps_per_epoch, self.epoch_size)
                return scheduler

            case "step_decay":
                scheduler = get_step_decay(lr, steps_per_epoch, self.epoch_size)
                return scheduler

        return None

    def __get_callbacks(self):
        callbacks_info = self.train_info['callbacks']

        # TensorBoard callback
        tensorboard_callback = get_tensorboard_callback(callbacks_info)
        callbacks = [tensorboard_callback]

        # ModelCheckpoint callbacks
        model_checkpoint_callbacks = get_model_checkpoint_callbacks(callbacks_info)
        callbacks.extend(model_checkpoint_callbacks)

        return callbacks

    def train_model(self):
        self.__save_train_config()

        # Continue training the model using loaded weights
        # and the specified epoch only if both are provided
        self.__train_from_weights()

        # Create datasets
        train_dataset, _ = self.__get_train_dataset()
        val_dataset, val_image_shape = self.__get_val_dataset()

        # Compile model
        loss = self.__get_loss()
        learning_rate = self.__get_learning_rate(
            steps_per_epoch=len(train_dataset))

        self.predictor.compile(
            loss=loss,
            learning_rate=learning_rate)

        # Callbacks
        validation_callback = ValidationCallback(
            val_dataset,
            image_shape=val_image_shape,
            input_shape=self.predictor.input_shape)

        # Train model
        self.predictor.fit(
            train_dataset,
            epochs=self.initial_epoch + self.epoch_size,
            initial_epoch=self.initial_epoch,
            callbacks=[validation_callback] + self.__get_callbacks())

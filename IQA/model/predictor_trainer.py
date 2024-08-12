import keras.losses
import pandas as pd

from model.predictor import Predictor
from util.callbacks.model_checkpoint_callbacks import get_model_checkpoint_callbacks
from util.callbacks.tensorboard_callback import get_tensorboard_callback
from util.callbacks.validation_callback import ValidationCallback
from util.schedulers.exponential_decay import get_exponential_decay
from util.schedulers.step_decay import get_step_decay
from util.flow_datasets_tf import flow_train_set_from_dataframe, flow_validation_set_from_dataframe


class PredictorTrainer:
    def __init__(self, train_info, model: Predictor):
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
        self.crop_image = self.train_info.get('crop_image')

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
                scheduler = get_exponential_decay(lr, steps_per_epoch, self.epoch_size)
                return scheduler

            case "step_decay":
                scheduler = get_step_decay(lr, steps_per_epoch, self.epoch_size)
                return scheduler

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

        # TensorBoard callback
        tensorboard_callback = get_tensorboard_callback(callbacks_info)
        callbacks = [tensorboard_callback]

        # ModelCheckpoint callbacks
        model_checkpoint_callbacks = get_model_checkpoint_callbacks(callbacks_info)
        callbacks.extend(model_checkpoint_callbacks)

        return callbacks

    def __get_train_dataset(self, dataframe, target_size):
        return flow_train_set_from_dataframe(
            dataframe,
            self.train_directory,
            self.batch_size,
            crop_size=target_size,
            augment=self.augment)

    def __get_val_dataset(self, dataframe, target_size):
        return flow_validation_set_from_dataframe(
            dataframe,
            self.val_directory,
            self.batch_size,
            crop_size=target_size)

    def train_model(self):
        # Continue training the model using loaded weights
        # and the specified epoch only if both are provided
        initial_epoch = self.continue_train.get('from_epoch')
        weights = self.continue_train.get('from_weights', '')

        if initial_epoch and weights:
            self.model.load_weights(weights)

        target_size = self.model.input_shape if self.crop_image else None

        # Create datasets
        train_df = pd.read_csv(self.train_lb)
        val_df = pd.read_csv(self.val_lb)
        train_dataset = self.__get_train_dataset(train_df, target_size)
        val_dataset = self.__get_val_dataset(val_df, target_size)

        # Compile model
        loss = self.__get_loss()
        learning_rate = self.__get_learning_rate(steps_per_epoch=len(train_dataset))

        self.model.compile(
            loss=loss,
            learning_rate=learning_rate)

        callbacks = self.__callbacks()

        # Use a validation callback if the dataset images are not
        # the same size as the model input
        if self.crop_image:
            validation_callback = ValidationCallback(
                data=val_dataset,
                loss=loss,
                target_size=target_size)

            # Insert the validation callback at the beginning,
            # so it will be called first
            callbacks = [validation_callback] + callbacks

        return self.model.fit(
            train_dataset,
            epochs=self.epoch_size + initial_epoch,
            initial_epoch=initial_epoch,
            validation_data=val_dataset if not self.crop_image else None,
            callbacks=callbacks)

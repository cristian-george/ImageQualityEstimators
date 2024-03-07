import os
import pandas as pd
import tensorflow as tf
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing import image

from models.metrics import srcc, plcc, rmse, emd
from models.resnet50.scheduler import LRSchedule
from util.random_crop import crop_generator
from util.gpu_utils import check_gpu_support, limit_gpu_memory, increase_cpu_num_threads


class ResNet50IQA:
    def __init__(self):
        self.image_original_size = (512, 384)
        self.model_input_shape = (384, 256)
        self.batch_size = 10
        self.epochs = 100
        self.model = self.build_model()

    def build_model(self):
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=(self.model_input_shape[0], self.model_input_shape[1], 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(2048, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.25)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(1, activation='linear')(x)
        return Model(inputs=base_model.input, outputs=output)

    def compile_model(self, total_batches):
        lr_schedule = LRSchedule(
            decay_epochs=[25, 40, 55, 70],
            learning_rates=[1e-4, 7.5e-5, 5e-5],
            total_epochs=self.epochs,
            total_batches=total_batches)
        self.model.compile(optimizer=Adam(learning_rate=lr_schedule),
                           loss=tf.keras.losses.Huber(),
                           metrics=[srcc, plcc, rmse, 'mae', emd])

    def train_model(self, train_dir, val_dir, train_labels_file, val_labels_file):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           horizontal_flip=True,
                                           rotation_range=15,
                                           fill_mode='reflect')
        val_datagen = ImageDataGenerator(rescale=1. / 255)
        train_df = pd.read_csv(train_labels_file)
        val_df = pd.read_csv(val_labels_file)
        train_generator = train_datagen.flow_from_dataframe(
            train_df, directory=train_dir, x_col='image_name', y_col='MOS',
            target_size=self.image_original_size, batch_size=self.batch_size, class_mode='raw')
        crop_train = crop_generator(train_generator, self.model_input_shape)
        validation_generator = val_datagen.flow_from_dataframe(
            val_df, directory=val_dir, x_col='image_name', y_col='MOS',
            target_size=self.image_original_size, batch_size=self.batch_size, class_mode='raw')
        crop_val = crop_generator(validation_generator, self.model_input_shape)
        tensorboard_callback = TensorBoard(log_dir='../../logs', histogram_freq=1)
        ckpt_dir = '../../trained_models/resnet50_finetune_huber_crop384x256'
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        best_model_callback = ModelCheckpoint(os.path.join(ckpt_dir, 'best_model.h5'),
                                              monitor='val_plcc',
                                              mode='max',
                                              save_best_only=True,
                                              save_weights_only=True)
        train_steps_per_epoch = train_generator.samples // self.batch_size
        self.compile_model(train_steps_per_epoch)
        self.model.fit(
            crop_train,
            steps_per_epoch=train_steps_per_epoch,
            epochs=self.epochs,
            validation_data=crop_val,
            validation_steps=validation_generator.samples // self.batch_size,
            callbacks=[tensorboard_callback, best_model_callback])

    def evaluate_model(self, test_dir, test_labels_file):
        test_df = pd.read_csv(test_labels_file)
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        test_generator = test_datagen.flow_from_dataframe(
            test_df, directory=test_dir, x_col='image_name', y_col='MOS',
            target_size=self.image_original_size, batch_size=self.batch_size, class_mode='raw')
        crop_test = crop_generator(test_generator, self.model_input_shape)
        val_loss = self.model.evaluate(
            crop_test,
            steps=test_generator.samples // self.batch_size)
        print(f'Values (huber, srcc, plcc, rmse, mae, emd): {val_loss}')

    def predict_score_for_image(self, image_path):
        img = image.load_img(image_path, target_size=self.model_input_shape)

        img_array = image.img_to_array(img)
        img_array /= 255.0

        img_tensor = tf.expand_dims(img_array, axis=0)

        score = self.model.predict(img_tensor, verbose=0)

        return score[0][0]


if __name__ == "__main__":
    use_gpu = check_gpu_support()
    if use_gpu:
        limit_gpu_memory(memory_limit=3500)
    else:
        increase_cpu_num_threads(num_threads=os.cpu_count())

    trainer = ResNet50IQA()
    trainer.model.summary()

    data_directory = '../../data/KonIQ-10K/'
    train_directory = data_directory + 'train/all_classes'
    val_directory = data_directory + 'val/all_classes'
    train_lb = data_directory + 'train_labels.csv'
    val_lb = data_directory + 'val_labels.csv'

    trainer.model.load_weights('../../trained_models/resnet50_finetune_huber_crop384x256/best_model.h5')
    trainer.compile_model(0)

    # trainer.train_model(train_directory, val_directory, train_lb, val_lb)

    test_directory = data_directory + 'test/all_classes'
    test_lb = data_directory + 'test_labels.csv'

    trainer.evaluate_model(test_directory, test_lb)

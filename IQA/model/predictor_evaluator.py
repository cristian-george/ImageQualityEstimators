import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from config_parser.evaluate_config_parser import EvaluateConfigParser
from model.predictor import Predictor
from util.preprocess_datasets import create_test_set_pipeline
from util.metrics import evaluate_metrics


def get_eval_file(test_directory, weights_path):
    dataset_name = test_directory.split("/")[-2]
    test_dir_name = test_directory.split("/")[-1]
    model_name = weights_path.split("/")[-1].split(".")[0]
    file_name = f'eval_{model_name}_{dataset_name}_{test_dir_name}.csv'

    weights_directory = "/".join(weights_path.split("/")[:-1])
    file_path = os.path.join(weights_directory, file_name)
    return file_name, file_path


class PredictorEvaluator:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor

        self.config_parser = EvaluateConfigParser()
        self.evaluate_info = self.config_parser.parse()

        self.__init_evaluate_info()

    def __init_evaluate_info(self):
        self.root_directory = self.evaluate_info['root_directory']
        self.test_directory = self.evaluate_info['test_directory']
        self.test_lb = self.evaluate_info['test_lb']

        self.weights_path = self.evaluate_info['weights_path']
        self.batch_size = self.evaluate_info['batch_size']

    def __get_test_dataset(self):
        test_df = pd.read_csv(self.test_lb)

        return create_test_set_pipeline(
            test_df,
            self.test_directory,
            self.batch_size,
            self.predictor.input_shape)

    def __predict_scores(self):
        # Create dataset
        dataset, image_shape = self.__get_test_dataset()
        target_size = self.predictor.input_shape

        # Predict scores
        predicted_scores = []
        for images, _ in tqdm(dataset, total=len(dataset), desc='Predict scores'):
            batch_size = tf.shape(images)[0]
            if image_shape != target_size:
                patches = tf.reshape(images, (-1,) + target_size)
                patch_predictions = self.predictor.predict(patches, batch_size=batch_size)
                patch_predictions = tf.reshape(patch_predictions, (batch_size, 5))
                patch_predictions = tf.reduce_mean(patch_predictions, axis=1)
                predicted_scores.append(patch_predictions.numpy())
            else:
                predictions = self.predictor.predict(images, batch_size=batch_size)
                predictions = tf.reshape(predictions, batch_size)
                predicted_scores.append(predictions.numpy())

        predicted_scores = np.concatenate(predicted_scores, axis=0)
        return predicted_scores

    @staticmethod
    def __evaluate_existing_file(file_path):
        df = pd.read_csv(file_path)

        y_true = df['true_MOS']
        y_pred = df['pred_MOS']

        PLCC, SRCC, MAE, RMSE = evaluate_metrics(y_true, y_pred)
        print("PLCC, SRCC, MAE, RMSE: ", PLCC, SRCC, MAE, RMSE)

    def __evaluate_new_file(self, file_path):
        test_df = pd.read_csv(self.test_lb)

        image_names = test_df['image_name']
        y_true = test_df['MOS']
        y_pred = self.__predict_scores()

        PLCC, SRCC, MAE, RMSE = evaluate_metrics(y_true, y_pred)
        print("PLCC, SRCC, MAE, RMSE: ", PLCC, SRCC, MAE, RMSE)

        # Store data in the file
        pd.DataFrame({
            'image_name': image_names,
            'true_MOS': y_true,
            'pred_MOS': y_pred
        }).to_csv(file_path, index=False)

    def evaluate_model(self):
        # Load weights and compile model
        self.predictor.load_weights(self.weights_path)
        self.predictor.compile()

        # Get path to the file which contains the predicted and true scores
        # for every image in the dataset marked for evaluation in evaluate_config
        file_name, file_path = get_eval_file(self.test_directory, self.weights_path)

        # Choose whether to override the existing file with new predicted values
        if os.path.isfile(file_path):
            override = input(f'File {file_name} already exists. '
                             f'Do you want to override it? (y/n): ')

            if override.lower() != 'y':
                print('File not overridden. Evaluate metrics...')
                self.__evaluate_existing_file(file_path)
                return

        # Predict scores for every image existing in dataset
        print(f'Evaluate metrics for {file_name}...')
        self.__evaluate_new_file(file_path)

    def evaluate_method(self, path, method='brisque'):
        test_df = pd.read_csv(self.test_lb)
        method_df = pd.read_csv(path)

        dataframe = pd.merge(test_df,
                             method_df,
                             on='image_name',
                             how='inner')

        real_scores = dataframe['MOS'].values
        method_scores = dataframe[method].values

        PLCC, SRCC, MAE, RMSE = evaluate_metrics(real_scores, method_scores)
        print("PLCC, SRCC, MAE, RMSE: ", PLCC, SRCC, MAE, RMSE)

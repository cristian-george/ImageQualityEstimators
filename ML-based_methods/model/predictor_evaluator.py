import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from config_parser.evaluate_config_parser import EvaluateConfigParser
from model.predictor import Predictor
from util.metrics import compute_metrics, compute_mean_std
from util.tf_dataset_pipeline import create_dataset_pipeline


def get_file(test_directory, weights_path):
    model_name = os.path.splitext(os.path.basename(weights_path))[0]
    dataset_name = os.path.basename(os.path.dirname(test_directory))
    test_dir_name = os.path.basename(test_directory)

    file_name = f'eval_{model_name}_{dataset_name}_{test_dir_name}.csv'
    weights_directory = os.path.dirname(weights_path)
    file_path = os.path.join(weights_directory, file_name)
    return file_name, file_path


class PredictorEvaluator:
    def __init__(self, predictor: Predictor):
        self.predictor = predictor
        self.config_parser = EvaluateConfigParser()
        self.evaluate_info = self.config_parser.parse()

    def __get_test_dataset(self, test_df, test_directory):
        return create_dataset_pipeline(
            test_df,
            test_directory,
            subset='test',
            net_name=self.predictor.net_name,
            batch_size=self.evaluate_info['batch_size'],
            target_size=self.predictor.input_shape
        )

    def __predict_scores(self, dataset, image_shape):
        input_shape = self.predictor.input_shape
        scores = []
        for batch in tqdm(dataset, total=len(dataset), desc='Predict scores'):
            images = batch[0] if isinstance(batch, tuple) else batch
            batch_size = tf.shape(images)[0]

            if image_shape == input_shape:
                predictions = tf.squeeze(self.predictor.predict(images, batch_size=batch_size), axis=1)
            else:
                patches = tf.reshape(images, (-1,) + input_shape)
                patch_predictions = self.predictor.predict(patches, batch_size=batch_size)
                patch_predictions = tf.reshape(patch_predictions, (batch_size, 5))
                predictions = tf.reduce_mean(patch_predictions, axis=1)

            scores.append(predictions.numpy())

        return np.concatenate(scores, axis=0)

    def evaluate_model(self):
        weights_path = self.evaluate_info['weights_path']
        test_dirs = self.evaluate_info['test_dirs']
        test_lbs = self.evaluate_info['test_lbs']

        self.predictor.load_weights(weights_path)
        self.predictor.compile()

        for idx, test_dir in enumerate(test_dirs):
            test_lb = test_lbs[idx] if test_lbs else None
            file_name, file_path = get_file(test_dir, weights_path)
            print(f"--- Evaluating {test_dir} on {self.predictor.net_name} ---")

            if os.path.isfile(file_path):
                override = input(f"File {file_name} already exists. Override? (y/n): ").strip().lower()
                if override != 'y':
                    print("File not overridden. Loading existing results.")
                    df = pd.read_csv(file_path)
                    if 'true_MOS' not in df.columns:
                        print("'true_MOS' not found. Only stats will be printed.")
                        compute_mean_std(df['pred_MOS'])
                    else:
                        compute_metrics(df['true_MOS'], df['pred_MOS'])
                    continue

            print(f"Evaluating and saving results to {file_name}...")
            self.__evaluate_and_save(test_dir, test_lb, file_path)

    @staticmethod
    def __load_test_dataframe(test_directory, test_lb=None):
        if test_lb:
            return pd.read_csv(test_lb)

        def list_images(directory):
            return sorted([
                f for f in os.listdir(directory)
                if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))
            ])

        image_names = list_images(test_directory)
        if not image_names:
            raise ValueError(f"No images found in {test_directory}")
        return pd.DataFrame({'image_name': image_names})

    def __evaluate_and_save(self, test_directory, test_lb, file_path):
        test_df = self.__load_test_dataframe(test_directory, test_lb)
        dataset, image_shape = self.__get_test_dataset(test_df, test_directory)
        y_pred = self.__predict_scores(dataset, image_shape)
        image_names = test_df['image_name']

        if 'MOS' in test_df.columns:
            y_true = test_df['MOS']
            compute_metrics(y_true, y_pred)
            df = pd.DataFrame({'image_name': image_names, 'true_MOS': y_true, 'pred_MOS': y_pred})
        else:
            print("'MOS' not found. Only predicted stats will be printed.")
            compute_mean_std(y_pred)
            df = pd.DataFrame({'image_name': image_names, 'pred_MOS': y_pred})

        df.to_csv(file_path, index=False)

import os

import keras.losses
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from model.model_class import ImageQualityPredictor
from util.correlations import srcc, plcc
from util.dataset_funcs import flow_test_set_from_dataframe


def evaluate_metrics(y_pred, y_true):
    PLCC = np.round(plcc(y_pred, y_true), 3)
    SRCC = np.round(srcc(y_pred, y_true), 3)

    MAE = np.round(np.mean(np.abs(y_pred - y_true)), 3)
    RMSE = np.round(np.sqrt(np.mean((y_pred - y_true) ** 2)), 3)

    print("PLCC, SRCC, MAE, RMSE: ", PLCC, SRCC, MAE, RMSE)


class PredictorEvaluator:
    def __init__(self, evaluate_info, model: ImageQualityPredictor):
        self.evaluate_info = evaluate_info
        self.model = model

        self.__init_evaluate_info()

    def __init_evaluate_info(self):
        self.root_directory = self.evaluate_info.get('root_directory', '')
        self.test_directory = self.evaluate_info.get('test_directory', '')
        self.test_lb = self.root_directory + self.evaluate_info.get('test_lb', '')
        self.weights_path = self.evaluate_info.get('weights_path', '')

        self.crop_image = self.evaluate_info.get('crop_image')
        self.batch_size = self.evaluate_info.get('batch_size')

    def __get_eval_file(self):
        dataset = self.root_directory.split("/")[-2]
        weights_dir = "/".join(self.weights_path.split("/")[:-1])
        model_name = self.weights_path.split("/")[-1].split(".")[0]

        file_name = f'eval_{model_name}_{dataset}_{self.test_directory}.csv'
        file_path = os.path.join(weights_dir, file_name)

        return file_name, file_path

    def __get_loss(self):
        loss = self.evaluate_info.get('loss', {})
        name = loss.get('name')

        match name:
            case 'huber':
                delta = loss.get('delta')
                return keras.losses.Huber(delta=delta)
            case 'mse':
                return keras.losses.MeanSquaredError()

    def __get_true_pred(self):
        file_name, file_path = self.__get_eval_file()

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Evaluation file {file_name} not found.")

        dataframe = pd.read_csv(file_path)
        y_true = dataframe['true_MOS']
        y_pred = dataframe['pred_MOS']

        return y_true, y_pred

    def predict_scores(self):
        test_df = pd.read_csv(self.test_lb)
        target_size = self.model.input_shape if self.crop_image else None
        dataset = flow_test_set_from_dataframe(test_df,
                                               self.root_directory + "/" + self.test_directory,
                                               batch_size=self.batch_size,
                                               target_size=target_size)

        image_names = test_df['image_name']
        y_true = test_df['MOS']
        y_pred = []

        for images, _ in tqdm(dataset, total=len(dataset), desc='Predict scores'):
            if self.crop_image:
                patches = tf.reshape(images, (-1, target_size[0], target_size[1], 3))
                patch_predictions = self.model.predict(patches, batch_size=self.batch_size)
                patch_predictions = tf.reshape(patch_predictions, (self.batch_size, 5))
                patch_predictions = tf.reduce_mean(patch_predictions, axis=1)
                y_pred.append(patch_predictions.numpy())
            else:
                predictions = self.model.predict(images, batch_size=self.batch_size)
                predictions = tf.reshape(predictions, self.batch_size)
                y_pred.append(predictions.numpy())

        y_pred = np.concatenate(y_pred, axis=0)

        return image_names, y_true, y_pred

    def evaluate_model(self):
        self.model.load_weights(self.weights_path)
        loss_fn = self.__get_loss()
        self.model.compile(loss=loss_fn)

        file_name, file_path = self.__get_eval_file()

        if os.path.isfile(file_path):
            override = input(f'File {file_name} already exists. '
                             f'Do you want to override it? (y/n): ')

            if override.lower() != 'y':
                print('File not overridden. Evaluate metrics...')
                dataframe = pd.read_csv(file_path)
                y_true = dataframe['true_MOS']
                y_pred = dataframe['pred_MOS']

                evaluate_metrics(y_pred, y_true)
                return

        image_names, y_true, y_pred = self.predict_scores()

        evaluate_metrics(y_pred, y_true)

        print('Bad predicted images among with their score and error: ')
        for i in range(len(y_pred)):
            if y_pred[i] < 1.0 or y_pred[i] > 5.0:
                print(f'Image name: {image_names[i]}, '
                      f'True score: {y_true[i]}. '
                      f'Predicted score: {y_pred[i]}')

        dataframe = pd.DataFrame({
            'image_name': image_names,
            'true_MOS': y_true,
            'pred_MOS': y_pred,
        })

        dataframe.to_csv(file_path, index=False)

    def evaluate_method(self, path, method='brisque'):
        df = pd.read_csv(self.test_lb)
        method_df = pd.read_csv(path)

        merged_df = pd.merge(df, method_df, on='image_name', how='inner')
        # merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

        mos_scores = merged_df['MOS'].values
        method_scores = merged_df[method].values
        evaluate_metrics(method_scores, mos_scores)

    def plot_prediction(self):
        try:
            y_true, y_pred = self.__get_true_pred()
            plt.scatter(y_true, y_pred, s=10, alpha=0.5)
            plt.xlabel('True Scores')
            plt.ylabel('Predicted Scores')
            plt.show()
        except FileNotFoundError as e:
            print(e)

    def plot_errors(self, function, title):
        try:
            y_true, y_pred = self.__get_true_pred()

            errors = function(y_pred, y_true)

            plt.scatter(y_true, errors, s=10, alpha=0.5)
            plt.xlabel('True Scores')
            plt.ylabel('Error')
            plt.title(title)
            plt.show()
        except FileNotFoundError as e:
            print(e)

    def plot_score_distribution(self):
        dataframe = pd.read_csv(self.test_lb)
        scores = dataframe['MOS']

        bins = list(np.arange(1.0, 5.05, 0.05))
        counts, bin_edges = np.histogram(scores, bins=bins)
        counts = counts / counts.max() * 0.9

        fig, ax = plt.subplots()
        ax.hist(bin_edges[:-1], bin_edges, weights=counts)

        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Scores distribution')
        ax.set_ylim(0.0, 1.0)

        fig_path = self.root_directory + "/score_distribution.svg"
        fig.savefig(fig_path, format='svg')

        plt.show()
        plt.close(fig)

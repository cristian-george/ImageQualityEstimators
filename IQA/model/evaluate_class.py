import os

import keras.losses
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from keras.applications.resnet import preprocess_input
from keras_preprocessing import image
from matplotlib import pyplot as plt

from util.correlations import srcc, plcc
from model.model_class import ImageQualityPredictor
from util.crop_funcs import crop_5patches


def evaluate_metrics(y_pred, y_true):
    PLCC_test = np.round(plcc(y_pred, y_true), 3)
    SRCC_test = np.round(srcc(y_pred, y_true), 3)

    MAE_test = np.round(np.mean(np.abs(y_pred - y_true)), 3)
    RMSE_test = np.round(np.sqrt(np.mean((y_pred - y_true) ** 2)), 3)

    print("PLCC, SRCC, MAE, RMSE: ", PLCC_test, SRCC_test, MAE_test, RMSE_test)


class PredictorEvaluator:
    def __init__(self, evaluate_info, model: ImageQualityPredictor):
        self.evaluate_info = evaluate_info
        self.model = model

        self.__init_evaluate_info()

    def __init_evaluate_info(self):
        self.root_directory = self.evaluate_info.get('root_directory', '')
        self.test_directory = self.root_directory + self.evaluate_info.get('test_directory', '')
        self.test_lb = self.root_directory + self.evaluate_info.get('test_lb', '')
        self.weights_path = self.evaluate_info.get('weights_path', '')
        self.test_scores = self.root_directory + self.evaluate_info.get('test_lb', '')

    def __get_loss(self):
        loss = self.evaluate_info.get('loss', {})
        name = loss.get('name')

        match name:
            case 'huber':
                delta = loss.get('delta')
                return keras.losses.Huber(delta=delta)
            case 'mse':
                return name

    def __predict_image_quality(self, image_path):
        img = image.load_img(image_path)
        img_array = image.img_to_array(img)

        if img_array.shape[:2] != self.model.input_shape:
            patches = crop_5patches(img_array, self.model.input_shape)
            scores = []
            for patch in patches:
                patch = preprocess_input(patch)
                crop_tensor = tf.expand_dims(patch, axis=0)
                score = self.model.predict(crop_tensor)
                scores.append(score)

            return np.average(scores)

        img_array = preprocess_input(img_array)
        img_tensor = tf.expand_dims(img_array, axis=0)
        score = self.model.predict(img_tensor)

        return score

    def __predict_quality(self, image_names):
        self.model.load_weights(self.weights_path)
        self.model.compile(self.__get_loss())

        predicted_scores = []
        for image_name in tqdm.tqdm(image_names, desc="Predict scores"):
            image_path = self.test_directory + "/" + image_name

            predicted_score = self.__predict_image_quality(image_path)
            predicted_scores.append(predicted_score)
        return predicted_scores

    def __get_data(self):
        df = pd.read_csv(self.test_scores)

        image_names = df['image_name'].values
        scores_MOS = df['MOS'].values

        return image_names, scores_MOS

    def evaluate_model(self):
        dataset = self.root_directory.split("/")[-2]
        weights_dir = "/".join(self.weights_path.split("/")[:-1])
        model_name = self.weights_path.split("/")[-1].split(".")[0]

        eval_file_name = f'eval_{model_name}_{dataset}_{self.test_directory}.csv'
        eval_file_path = os.path.join(weights_dir, eval_file_name)

        if os.path.isfile(eval_file_path):
            override = input(f'File {eval_file_name} already exists. '
                             f'Do you want to override it? (y/n): ')
            if override.lower() != 'y':
                print('File not overridden. Evaluate metrics...')
                df = pd.read_csv(eval_file_path)
                y_true = df['true_MOS']
                y_pred = df['pred_MOS']

                evaluate_metrics(y_pred, y_true)
                return

        image_names, y_true = self.__get_data()

        y_pred = self.__predict_quality(image_names)
        y_pred = np.array(y_pred)

        df = pd.DataFrame({
            'image_name': image_names,
            'true_MOS': y_true,
            'pred_MOS': y_pred,
        })
        df.to_csv(eval_file_path, index=False)

        evaluate_metrics(y_pred, y_true)

    def evaluate_method(self, path, method='brisque'):
        df = pd.read_csv(self.test_scores)
        method_df = pd.read_csv(path)

        merged_df = pd.merge(df, method_df, on='image_name', how='inner')
        # merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

        mos_scores = merged_df['MOS'].values
        method_scores = merged_df[method].values
        evaluate_metrics(method_scores, mos_scores)

    def __get_true_pred(self):
        dataset = self.root_directory.split("/")[-2]
        weights_dir = "/".join(self.weights_path.split("/")[:-1])
        model_name = self.weights_path.split("/")[-1].split(".")[0]

        eval_file_name = f'eval_{model_name}_{dataset}_{self.test_directory}.csv'
        eval_file_path = os.path.join(weights_dir, eval_file_name)

        if not os.path.isfile(eval_file_path):
            print(f"Evaluation file {eval_file_name} is not found.")
            return

        df = pd.read_csv(eval_file_path)
        image_names = df['image_name']
        y_true = df['true_MOS']
        y_pred = df['pred_MOS']

        # Print bad predicted images among with their score and error
        for i in range(len(y_pred)):
            if y_pred[i] < 1 or y_pred[i] > 5:
                print(
                    f"Image Name: {image_names[i]}, "
                    f"Predicted Score: {y_pred[i]}, "
                    f"True Score: {y_true[i]}")

        return y_true, y_pred

    def plot_prediction(self):
        y_test, y_pred = self.__get_true_pred()

        plt.scatter(y_test, y_pred, s=10, alpha=0.5)
        plt.xlabel('True Scores')
        plt.ylabel('Predicted Scores')
        plt.show()

    def plot_errors(self, function, title):
        y_test, y_pred = self.__get_true_pred()

        errors = function(y_pred, y_test)

        plt.scatter(y_test, errors, s=10, alpha=0.5)
        plt.xlabel('True Scores')
        plt.ylabel('Error')
        plt.title(title)
        plt.show()

    def plot_score_distribution(self):
        _, scores = self.__get_data()

        bins = list(np.arange(1.0, 5.05, 0.05))
        counts, bin_edges = np.histogram(scores, bins=bins)
        counts = counts / counts.max() * 0.9

        fig, ax = plt.subplots()
        ax.hist(bin_edges[:-1], bin_edges, weights=counts)

        ax.set_xlabel('Score Range')
        ax.set_ylabel('Normalized Frequencies')
        ax.set_title('Histogram of Scores')
        ax.set_ylim(0.0, 1.0)

        fig_path = self.root_directory + "/score_distribution.svg"
        fig.savefig(fig_path, format='svg')

        plt.show()
        plt.close(fig)

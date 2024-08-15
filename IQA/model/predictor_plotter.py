import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from model.predictor_evaluator import get_eval_file


class PredictorPlotter:
    def __init__(self, evaluate_info):
        self.evaluate_info = evaluate_info

        self.__init_evaluate_info()

    def __init_evaluate_info(self):
        self.root_directory = self.evaluate_info.get('root_directory', '')
        self.test_directory = self.evaluate_info.get('test_directory', '')
        self.test_lb = self.root_directory + self.evaluate_info.get('test_lb', '')
        self.weights_path = self.evaluate_info.get('weights_path', '')

    def __get_scores(self):
        file_name, file_path = get_eval_file(self.evaluate_info)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Evaluation file {file_name} not found.")

        df = pd.read_csv(file_path)
        y_true = df['true_MOS']
        y_pred = df['pred_MOS']

        return y_true, y_pred

    def plot_prediction(self):
        try:
            y_true, y_pred = self.__get_scores()
            plt.scatter(y_true, y_pred, s=10, alpha=0.5)
            plt.xlabel('True Scores')
            plt.ylabel('Predicted Scores')
            plt.show()
        except FileNotFoundError as e:
            print(e)

    def plot_errors(self, function, title):
        try:
            y_true, y_pred = self.__get_scores()

            errors = function(y_pred, y_true)

            plt.scatter(y_true, errors, s=10, alpha=0.5)
            plt.xlabel('True Scores')
            plt.ylabel('Error')
            plt.title(title)
            plt.show()
        except FileNotFoundError as e:
            print(e)

    def plot_score_distribution(self):
        df = pd.read_csv(self.test_lb)
        scores = df['MOS']

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

import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config_parser.evaluate_config_parser import EvaluateConfigParser
from model.predictor_evaluator import get_eval_file


class PredictorPlotter:
    def __init__(self):
        self.config_parser = EvaluateConfigParser()
        self.evaluate_info = self.config_parser.parse()

        self.__init_evaluate_info()

    def __init_evaluate_info(self):
        self.root_directory = self.evaluate_info['root_directory']
        self.test_directory = self.evaluate_info['test_directory']
        self.test_lb = self.evaluate_info['test_lb']
        self.weights_path = self.evaluate_info['weights_path']

    def __get_scores(self):
        file_name, file_path = get_eval_file(self.test_directory, self.weights_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Evaluation file {file_name} not found.")

        df = pd.read_csv(file_path)
        y_true = df['true_MOS']
        y_pred = df['pred_MOS']

        return y_true, y_pred

    def plot_prediction(self):
        y_true, y_pred = self.__get_scores()

        fig, ax = plt.subplots()

        ax.scatter(y_true, y_pred, s=10, alpha=0.5)
        ax.set_xlabel('True score', fontsize=13)
        ax.set_ylabel('Predicted score', fontsize=13)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.subplots_adjust(top=0.975, right=0.975)

        fig_path = self.root_directory + (f'/prediction'
                                          f'_{self.test_directory.split("/")[-2]}'  # dataset name
                                          f'_{self.test_directory.split("/")[-1]}'  # test directory
                                          f'.svg')
        fig.savefig(fig_path,
                    format='svg')

        plt.show()
        plt.close(fig)

    def plot_errors(self, function, title):
        y_true, y_pred = self.__get_scores()
        errors = function(y_true, y_pred)

        fig, ax = plt.subplots()

        ax.scatter(y_true, errors, s=10, alpha=0.5)
        ax.set_xlabel('True score', fontsize=13)
        ax.set_ylabel(title + ' error', fontsize=13)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.subplots_adjust(top=0.975, right=0.975)

        fig_path = self.root_directory + (f'/{title}_error'
                                          f'_{self.test_directory.split("/")[-2]}'  # dataset name
                                          f'_{self.test_directory.split("/")[-1]}'  # test directory
                                          f'.svg')

        fig.savefig(fig_path,
                    format='svg')

        plt.show()
        plt.close(fig)

    def plot_score_distribution(self):
        df = pd.read_csv(self.test_lb)
        scores = df['MOS']

        bins = list(np.arange(1.0, 5.05, 0.05))
        counts, bin_edges = np.histogram(scores, bins=bins)
        counts = counts / counts.sum()

        fig, ax = plt.subplots()
        ax.hist(bin_edges[:-1], bin_edges, weights=counts)

        ax.set_xlabel('Quality score', fontsize=13)
        ax.set_ylabel('Relative frequency', fontsize=13)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.subplots_adjust(top=0.975, right=0.975)

        y_min, y_max = ax.get_ylim()
        ax.set_ylim(0.0, y_max + 0.01)

        fig_path = self.root_directory + (f'/score_distribution'
                                          f'_{self.test_directory.split("/")[-2]}'  # dataset name
                                          f'_{self.test_directory.split("/")[-1]}'  # test directory
                                          f'.svg')
        fig.savefig(fig_path,
                    format='svg')

        plt.show()
        plt.close(fig)

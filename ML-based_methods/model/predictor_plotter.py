import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config_parser.evaluate_config_parser import EvaluateConfigParser
from model.predictor_evaluator import get_file


class PredictorPlotter:
    def __init__(self):
        self.config_parser = EvaluateConfigParser()
        self.evaluate_info = self.config_parser.parse()
        self.__init_evaluate_info()

    def __init_evaluate_info(self):
        self.test_dirs = self.evaluate_info['test_dirs']
        self.test_lbs = self.evaluate_info['test_lbs']
        self.weights_path = self.evaluate_info['weights_path']

    def __get_scores(self, test_dir):
        file_name, file_path = get_file(test_dir, self.weights_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Evaluation file {file_name} not found.")

        df = pd.read_csv(file_path)
        y_true = df['true_MOS']
        y_pred = df['pred_MOS']

        return y_true, y_pred

    def plot_prediction(self):
        for test_dir in self.test_dirs:
            y_true, y_pred = self.__get_scores(test_dir)

            fig, ax = plt.subplots()
            ax.scatter(y_true, y_pred, s=10, alpha=0.5)
            ax.set_xlabel('True score', fontsize=20)
            ax.set_ylabel('Predicted score', fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.subplots_adjust(top=0.975, right=0.975, left=0.2, bottom=0.15)

            os.makedirs(os.path.dirname(self.weights_path) + f'/_prediction plots/', exist_ok=True)
            fig_path = os.path.dirname(self.weights_path) + (f'/_prediction plots/'
                                                             f'{os.path.basename(os.path.dirname(test_dir))}_'
                                                             f'{os.path.basename(test_dir)}.png')
            fig.savefig(fig_path, format='png', dpi=300)
            plt.show()
            plt.close(fig)

    def plot_errors(self, dir_name, function, label_x='MOS', label_y='Predicted MOS'):
        for test_dir in self.test_dirs:
            y_true, y_pred = self.__get_scores(test_dir)
            errors = function(y_true, y_pred)

            fig, ax = plt.subplots()
            ax.scatter(y_true, errors, s=10, alpha=0.5)
            ax.set_xlabel(label_x, fontsize=20)
            ax.set_ylabel(label_y, fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.subplots_adjust(top=0.975, right=0.975, left=0.2, bottom=0.15)
            ax.set_xlim(0.95, 5.05)

            os.makedirs(os.path.dirname(self.weights_path) + f'/_{dir_name} error plots/', exist_ok=True)
            fig_path = os.path.dirname(self.weights_path) + (f'/_{dir_name} error plots/'
                                                             f'{os.path.basename(os.path.dirname(test_dir))}_'
                                                             f'{os.path.basename(test_dir)}.png')
            fig.savefig(fig_path, format='png', dpi=300)
            plt.show()
            plt.close(fig)

    def plot_score_distribution(self):
        for test_dir, test_lb in zip(self.test_dirs, self.test_lbs):
            df = pd.read_csv(test_lb)
            scores = df['MOS']

            bins = list(np.arange(1.0, 5.05, 0.05))
            counts, bin_edges = np.histogram(scores, bins=bins)
            counts = counts / counts.sum()

            fig, ax = plt.subplots()
            ax.hist(bin_edges[:-1], bin_edges, weights=counts)
            ax.set_xlabel('Quality score', fontsize=20)
            ax.set_ylabel('Relative frequency', fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.subplots_adjust(top=0.975, right=0.975, left=0.2, bottom=0.15)

            y_min, y_max = ax.get_ylim()
            ax.set_ylim(0.0, y_max + 0.01)

            fig_path = os.path.join(
                os.path.dirname(test_dir),
                f'score_distribution_{os.path.basename(os.path.dirname(test_dir))}_{os.path.basename(test_dir)}.png'
            )
            fig.savefig(fig_path, format='png', dpi=300)
            plt.show()
            plt.close(fig)

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ModelPlotting:
    def __init__(self, model, evaluate_info):
        self.model = model
        self.root_directory = evaluate_info.get('root_directory', '')
        self.test_directory = evaluate_info.get('test_directory', '')
        self.weights_path = evaluate_info.get('weights_path', '')
        self.test_scores = self.root_directory + evaluate_info.get('test_lb', '')

    def __get_data(self):
        df = pd.read_csv(self.test_scores)

        image_names = df['image_name'].values
        score_MOS = df['MOS'].values

        return image_names, score_MOS

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

        # Plot the results
        plt.scatter(y_test, y_pred, s=10, alpha=0.5)
        plt.xlabel('True Scores')
        plt.ylabel('Predicted Scores')
        plt.show()

    def plot_errors(self, function, title):
        y_test, y_pred = self.__get_true_pred()

        # Calculate error for each image
        errors = function(y_pred, y_test)

        # Plot the results
        plt.scatter(y_test, errors, s=10, alpha=0.5)
        plt.xlabel('True Scores')
        plt.ylabel('Error')
        plt.title(title)
        plt.show()

    def plot_score_distribution(self):
        _, scores = self.__get_data()

        # Define the bins
        bins = list(np.arange(1.0, 5.05, 0.05))

        # Compute the histogram
        counts, bin_edges = np.histogram(scores, bins=bins)

        # Normalize the counts so that the highest bin reaches 0.9
        counts = counts / counts.max() * 0.9

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the normalized histogram
        ax.hist(bin_edges[:-1], bin_edges, weights=counts)

        # Set labels and title
        ax.set_xlabel('Score Range')
        ax.set_ylabel('Normalized Frequencies')
        ax.set_title('Histogram of Scores')

        # Set y-axis limits
        ax.set_ylim(0.0, 1.0)

        # Save the plot as an SVG file
        fig_path = self.root_directory + "/score_distribution.svg"
        fig.savefig(fig_path, format='svg')

        # Show the plot
        plt.show()

        # Close the plot to free memory
        plt.close(fig)

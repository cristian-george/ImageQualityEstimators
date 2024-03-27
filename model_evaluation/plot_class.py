import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ModelPlotting:
    def __init__(self, model, evaluate_info):
        self.model = model
        self.data_directory = evaluate_info.get('data_directory', '')
        self.test_scores = self.data_directory + evaluate_info.get('test_lb', '')

    def __get_true_scores(self):
        df = pd.read_csv(self.test_scores)

        scores = df['MOS'].values
        return scores

    def __get_data(self):
        df = pd.read_csv(self.test_scores)

        image_names = df['image_name'].values
        y_test = df['MOS'].values

        y_pred = self.model.predict_scores(image_names)
        y_pred = np.array(y_pred)

        # Print bad predicted images among with their score and error
        for i in range(len(y_pred)):
            if y_pred[i] < 1 or y_pred[i] > 5:
                print(
                    f"Image Name: {image_names[i]}, "
                    f"Predicted Score: {y_pred[i]}, "
                    f"True Score: {y_test[i]}")

        return y_test, y_pred

    def plot_prediction(self):
        y_test, y_pred = self.__get_data()

        # Plot the results
        plt.scatter(y_test, y_pred, s=10, alpha=0.5)
        plt.xlabel('True Scores')
        plt.ylabel('Predicted Scores')
        plt.show()

    def plot_errors(self, function, title):
        y_test, y_pred = self.__get_data()

        # Calculate error for each image
        errors = function(y_pred, y_test)

        # Plot the results
        plt.scatter(y_test, errors, s=10, alpha=0.5)
        plt.xlabel('True Scores')
        plt.ylabel('Error')
        plt.title(title)
        plt.show()

    def plot_score_distribution(self):
        scores = self.__get_true_scores()

        # Define the bins
        bins = [1, 2, 3, 4, 5]

        # Plot the histogram
        plt.hist(scores, bins=bins, edgecolor='black')

        # Set labels and title
        plt.xlabel('Score Range')
        plt.ylabel('Frequency')
        plt.title('Histogram of Scores')
        plt.show()

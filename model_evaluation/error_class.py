import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PlotErrors:
    def __init__(self, model, evaluate_info):
        self.model = model
        self.data_directory = evaluate_info.get('data_directory', '')
        self.test_scores = self.data_directory + evaluate_info.get('test_lb', '')

    def plot_errors(self, function, title):
        df = pd.read_csv(self.test_scores)

        image_names = df['image_name'].values
        y_test = df['MOS'].values

        y_pred = self.model.predict_scores(image_names)
        y_pred = np.array(y_pred)

        # Calculate error for each image
        errors = function(y_pred, y_test)

        # Plot the results
        plt.scatter(y_test, errors, s=10, alpha=0.5)
        plt.xlabel('True Scores')
        plt.ylabel('Error')
        plt.title(title)
        plt.show()

        # Print some example predicted and true scores along with errors
        for i in range(5):
            print(
                f"Image Name: {image_names[i]}, "
                f"Predicted Score: {y_pred[i]}, "
                f"True Score: {y_test[i]}, "
                f"Error: {errors[i]}")

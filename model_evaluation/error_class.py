import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PlotErrors:
    def __init__(self, model, evaluate_info):
        self.model = model
        self.data_directory = evaluate_info.get('data_directory', '')
        self.test_images = self.data_directory + evaluate_info.get('test_directory', '')
        self.test_scores = self.data_directory + evaluate_info.get('test_lb', '')
        self.weights_path = evaluate_info.get('weights_path', '')

    def plot_errors(self, function, title):
        df = pd.read_csv(self.test_scores)

        image_names = df['image_name'].values
        true_scores = df['MOS'].values

        predicted_scores = self.__predict_scores(image_names)
        predicted_scores = np.array(predicted_scores)

        # Calculate error for each image
        errors = function(predicted_scores, true_scores)

        # Plot the results
        plt.scatter(true_scores, errors, s=10, alpha=0.5)
        plt.xlabel('True Scores')
        plt.ylabel('Error')
        plt.title(title)
        plt.show()

        # Print some example predicted and true scores along with errors
        for i in range(5):
            print(
                f"Image Name: {image_names[i]}, "
                f"Predicted Score: {predicted_scores[i]}, "
                f"True Score: {true_scores[i]}, "
                f"Error: {errors[i]}")

    def __predict_scores(self, image_names):
        self.model.load_weights(self.weights_path)
        self.model.compile_model()

        predicted_scores = []
        for image_name in image_names:
            image_path = self.test_images + "/" + image_name

            predicted_score = self.model.predict_score_for_image(image_path)  # Predict score for the image
            predicted_scores.append(predicted_score)
        return predicted_scores

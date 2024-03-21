import pandas as pd
import numpy as np

from model.correlations import plcc, srcc


class ModelEvaluation:
    def __init__(self, model, evaluate_info):
        self.model = model
        self.data_directory = evaluate_info.get('data_directory', '')
        self.test_images = self.data_directory + evaluate_info.get('test_directory', '')
        self.test_scores = self.data_directory + evaluate_info.get('test_lb', '')
        self.weights_path = evaluate_info.get('weights_path', '')

    def evaluate(self):
        df = pd.read_csv(self.test_scores)

        image_names = df['image_name'].values
        y_test = df['MOS'].values

        y_pred = self.__predict_scores(image_names)
        y_pred = np.array(y_pred)

        PLCC_test = np.round(plcc(y_pred, y_test), 3)
        SRCC_test = np.round(srcc(y_pred, y_test), 3)

        print("PLCC, SRCC: ", PLCC_test, SRCC_test)

    def __predict_scores(self, image_names):
        self.model.load_weights(self.weights_path)
        self.model.compile_model()

        predicted_scores = []
        for image_name in image_names:
            image_path = self.test_images + "/" + image_name

            predicted_score = self.model.predict_score_for_image(image_path)  # Predict score for the image
            predicted_scores.append(predicted_score)
        return predicted_scores

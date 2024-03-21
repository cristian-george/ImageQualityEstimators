import pandas as pd
import numpy as np

from model.correlations import plcc, srcc


class ModelEvaluation:
    def __init__(self, model, evaluate_info):
        self.model = model
        self.data_directory = evaluate_info.get('data_directory', '')
        self.test_scores = self.data_directory + evaluate_info.get('test_lb', '')

    def evaluate(self):
        df = pd.read_csv(self.test_scores)

        image_names = df['image_name'].values
        y_test = df['MOS'].values

        y_pred = self.model.predict_scores(image_names)
        y_pred = np.array(y_pred)

        PLCC_test = np.round(plcc(y_pred, y_test), 3)
        SRCC_test = np.round(srcc(y_pred, y_test), 3)

        print("PLCC, SRCC: ", PLCC_test, SRCC_test)

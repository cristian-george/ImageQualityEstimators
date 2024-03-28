import pandas as pd
import numpy as np

from model.correlations import plcc, srcc


def evaluate_metrics(y_pred, y_true):
    PLCC_test = np.round(plcc(y_pred, y_true), 3)
    SRCC_test = np.round(srcc(y_pred, y_true), 3)

    MAE_test = np.round(np.mean(np.abs(y_pred - y_true)), 3)
    RMSE_test = np.round(np.sqrt(np.mean((y_pred - y_true) ** 2)), 3)

    print("PLCC, SRCC, MAE, RMSE: ", PLCC_test, SRCC_test, MAE_test, RMSE_test)


class ModelEvaluation:
    def __init__(self, model, evaluate_info):
        self.model = model
        self.data_directory = evaluate_info.get('data_directory', '')
        self.test_scores = self.data_directory + evaluate_info.get('test_lb', '')

    def __get_data(self):
        df = pd.read_csv(self.test_scores)

        image_names = df['image_name'].values
        score_MOS = df['MOS'].values

        return image_names, score_MOS

    def evaluate(self):
        image_names, y_true = self.__get_data()

        y_pred = self.model.predict_scores(image_names)
        y_pred = np.array(y_pred)

        evaluate_metrics(y_pred, y_true)

    def evaluate_method(self, path, method='brisque'):
        df = pd.read_csv(self.test_scores)
        method_df = pd.read_csv(path)

        merged_df = pd.merge(df, method_df, on='image_name', how='inner')
        # merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

        mos_scores = merged_df['MOS'].values
        method_scores = merged_df[method].values
        evaluate_metrics(method_scores, mos_scores)


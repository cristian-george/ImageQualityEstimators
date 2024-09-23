import pandas as pd

from util.metrics import evaluate_metrics

datasets = ['LIVE2', 'KonIQ-10K', 'LIVEitW', 'FLIVE']


def opencv_brisque():
    for dataset in datasets:
        method = dataset + '_opencv_brisque.csv'
        dataset_path = f'../../Datasets/{dataset}/'
        MOS_path = dataset_path + dataset + '_labels.csv'
        brisque_path = dataset_path + method

        test_df = pd.read_csv(MOS_path)
        method_df = pd.read_csv(brisque_path)

        dataframe = pd.merge(test_df,
                             method_df,
                             on='image_name')

        scores = dataframe['MOS'].values
        method_scores = dataframe['opencv_brisque'].values

        evaluate_method(scores, method_scores, method)


def matlab_brisque(trained_on='KonIQ-10K'):
    for dataset in datasets:
        matlab_brisque_helper(dataset, trained_on, 'images')

        if trained_on == dataset:
            matlab_brisque_helper(dataset, trained_on, 'test')


def matlab_brisque_helper(dataset, trained_on='KonIQ-10K', test_on='images'):
    method = trained_on + '_model_' + dataset + '_' + test_on + '_matlab_brisque.csv'
    dataset_path = f'../../Datasets/{dataset}/'
    MOS_path = dataset_path + dataset + '_labels.csv'
    brisque_path = dataset_path + method

    test_df = pd.read_csv(MOS_path)
    method_df = pd.read_csv(brisque_path)

    dataframe = pd.merge(test_df,
                         method_df,
                         on='image_name')

    scores = dataframe['MOS'].values
    method_scores = dataframe['matlab_brisque'].values

    evaluate_method(scores, method_scores, method)


def evaluate_method(y_true, y_pred, method):
    print(f"Evaluate: {method}")
    PLCC, SRCC, MAE, RMSE = evaluate_metrics(y_true, y_pred)
    print("SRCC, PLCC, RMSE, MAE: ", SRCC, PLCC, RMSE, MAE)


if __name__ == '__main__':
    # opencv_brisque()
    matlab_brisque('LIVE2')
    matlab_brisque('KonIQ-10K')

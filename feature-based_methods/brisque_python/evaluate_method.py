import pandas as pd

from util.metrics import compute_metrics

datasets = ['LIVE2', 'KonIQ-10K', 'LIVEitW', 'FLIVE']
biqa_matlab_estimators = ['niqe', 'brisque']
biqa_toolbox_estimators = ['biqi', 'bliinds', 'divine']


def opencv_brisque():
    for dataset in datasets:
        method_filename = dataset + '_opencv_brisque.csv'
        dataset_path = f'../../Datasets/{dataset}/'

        MOS_path = dataset_path + dataset + '_labels.csv'
        brisque_path = dataset_path + 'brisque_eval/' + method_filename

        test_df = pd.read_csv(MOS_path)
        method_df = pd.read_csv(brisque_path)

        dataframe = pd.merge(test_df,
                             method_df,
                             on='image_name')

        scores = dataframe['MOS'].values
        method_scores = dataframe['opencv_brisque'].values

        evaluate_method(scores, method_scores, method_filename)


def biqa_matlab_estimator(trained_on='KonIQ-10K'):
    for dataset in datasets:
        print(f"Evaluate BIQA estimators for {dataset}")
        for estimator in biqa_matlab_estimators:
            if trained_on != dataset:
                biqa_matlab_estimator_helper(dataset, estimator, trained_on, 'images')
            else:
                biqa_matlab_estimator_helper(dataset, estimator, trained_on, 'test')


def biqa_matlab_estimator_helper(dataset, estimator, trained_on, test_on):
    method_filename = trained_on + '_model_' + dataset + '_' + test_on + '_matlab_' + estimator + '.csv'
    dataset_path = f'../../Datasets/{dataset}/'

    MOS_path = dataset_path + dataset + '_labels.csv'
    estimator_path = dataset_path + f'{estimator}_eval/' + method_filename

    try:
        test_df = pd.read_csv(MOS_path)
        method_df = pd.read_csv(estimator_path)

        dataframe = pd.merge(test_df,
                             method_df,
                             on='image_name')

        scores = dataframe['MOS'].values
        method_scores = dataframe[f'matlab_{estimator}'].values

        evaluate_method(scores, method_scores, method_filename)

    except FileNotFoundError as e:
        print(f"File not found: {e}")


def biqa_toolbox_estimator():
    for dataset in datasets:
        print(f"Evaluate BIQA estimators for {dataset}")
        for estimator in biqa_toolbox_estimators:
            biqa_toolbox_estimator_helper(dataset, estimator)


def biqa_toolbox_estimator_helper(dataset, estimator):
    dataset_path = f'../../Datasets/{dataset}/'

    MOS_path = dataset_path + dataset + '_labels.csv'
    estimator_path = dataset_path + dataset + '_BIQA.csv'

    test_df = pd.read_csv(MOS_path)
    method_df = pd.read_csv(estimator_path)

    dataframe = pd.merge(test_df,
                         method_df,
                         on='image_name')

    scores = dataframe['MOS'].values
    biqa_scores = 5 - dataframe[estimator].values / 25

    evaluate_method(scores, biqa_scores, estimator)


def evaluate_method(y_true, y_pred, method):
    print(f"Evaluate: {method}")
    compute_metrics(y_true, y_pred, verbose=True)


if __name__ == '__main__':
    # opencv_brisque()
    biqa_matlab_estimator('LIVE')

    # biqa_matlab_estimator('LIVE2')
    # biqa_matlab_estimator('KonIQ-10K')

    # Inside each dataset folder there should be a file named "dataset_BIQA.csv" in order to evaluate BIQA estimators
    # biqa_toolbox_estimator()

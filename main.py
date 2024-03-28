import os

import numpy as np

from model_config.model_config import ConfigParser
from model.model_class import IQA
from model_evaluation.plot_class import ModelPlotting
from model_evaluation.evaluate_class import ModelEvaluation
from util.gpu_utils import check_gpu_support, limit_gpu_memory, increase_cpu_num_threads

if __name__ == "__main__":
    use_gpu = check_gpu_support()
    if use_gpu:
        limit_gpu_memory(memory_limit=3500)
    else:
        increase_cpu_num_threads(num_threads=os.cpu_count())

    while True:
        print("Options: ")
        print("1 for training a model")
        print("2 to evaluate a model batch by batch")
        print("3 to evaluate a model image by image")
        print("4 to plot distribution of scores in dataset")
        print("5 to plot predicted and true scores")
        print("6 to plot difference error between predicted and true scores")
        print("7 to plot absolute error between predicted and true scores")
        print("8 to evaluate BRISQUE")
        print("0 to exit")

        option = int(input("Enter option: "))
        if option == 0:
            break

        config_parser = ConfigParser('model_config/config.json')

        model = IQA(config_parser.get_model_info(),
                    config_parser.get_learn_info(),
                    config_parser.get_train_info(),
                    config_parser.get_evaluate_info())
        # model.model.summary(show_trainable=True)

        match option:
            case 1:
                config_parser.save_config_json()
                model.train_model()
            case 2:
                model.evaluate_model()
            case 3:
                model_eval = ModelEvaluation(model, config_parser.get_evaluate_info())
                model_eval.evaluate()
            case 4:
                plot_distribution = ModelPlotting(None, config_parser.get_evaluate_info())
                plot_distribution.plot_score_distribution()
            case 5:
                plot_pred = ModelPlotting(model, config_parser.get_evaluate_info())
                plot_pred.plot_prediction()
            case 6:
                plot_err = ModelPlotting(model, config_parser.get_evaluate_info())
                plot_err.plot_errors(lambda x, y: x - y, 'Difference (pred - true) vs. True Scores')
            case 7:
                plot_err = ModelPlotting(model, config_parser.get_evaluate_info())
                plot_err.plot_errors(lambda x, y: np.abs(x - y), 'Absolute Error vs. True Scores')
            case 8:
                model_eval = ModelEvaluation(model, config_parser.get_evaluate_info())
                model_eval.evaluate_method('data/LIVE2/LIVE2_matlab_brisque.csv',
                                           method='brisque')

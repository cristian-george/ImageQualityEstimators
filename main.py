import os

import numpy as np

from model_config.config_parser import ConfigParser
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
        print("1 for printing model summary")
        print("2 for training a model")
        print("3 to evaluate a model batch by batch")
        print("4 to evaluate a model image by image")
        print("5 to evaluate a model image by image - keep aspect ratio")
        print("6 to plot distribution of scores in dataset")
        print("7 to plot predicted and true scores")
        print("8 to plot difference error between predicted and true scores")
        print("9 to plot absolute error between predicted and true scores")
        # print("10 to evaluate BRISQUE")
        print("0 to exit")

        option = int(input("Enter option: "))
        if option == 0:
            break

        config_parser = ConfigParser()

        model = IQA(config_parser.get_model_info(),
                    config_parser.get_train_info(),
                    config_parser.get_evaluate_info())
        model.build_model()

        match option:
            case 1:
                model.summary()
            case 2:
                config_parser.save_train_config()
                model.train_model()
            case 3:
                model.evaluate_model()
            case 4:
                model_eval = ModelEvaluation(model, config_parser.get_evaluate_info())
                model_eval.evaluate(keep_aspect_ratio=False)
            case 5:
                model_eval = ModelEvaluation(model, config_parser.get_evaluate_info())
                model_eval.evaluate(keep_aspect_ratio=True)
            case 6:
                plot_distribution = ModelPlotting(None, config_parser.get_evaluate_info())
                plot_distribution.plot_score_distribution()
            case 7:
                plot_pred = ModelPlotting(model, config_parser.get_evaluate_info())
                plot_pred.plot_prediction()
            case 8:
                plot_err = ModelPlotting(model, config_parser.get_evaluate_info())
                plot_err.plot_errors(lambda x, y: x - y, 'Difference Error vs. True Scores')
            case 9:
                plot_err = ModelPlotting(model, config_parser.get_evaluate_info())
                plot_err.plot_errors(lambda x, y: np.abs(x - y), 'Absolute Error vs. True Scores')
            # case 10:
            #     model_eval = ModelEvaluation(model, config_parser.get_evaluate_info())
            #     model_eval.evaluate_method('data/LIVE2/LIVE2_matlab_brisque.csv',
            #                                method='brisque')

import os

import numpy as np

from model.predictor import Predictor
from model.predictor_evaluator import PredictorEvaluator
from model.predictor_plotter import PredictorPlotter
from model.predictor_trainer import PredictorTrainer
from util.gpu_tf import check_gpu_support, limit_gpu_memory, increase_cpu_num_threads

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    use_gpu = check_gpu_support()
    if use_gpu:
        limit_gpu_memory(memory_limit=10 * 1024)
    else:
        increase_cpu_num_threads(num_threads=os.cpu_count())

    while True:
        print("Options: ")
        print("1 to train a model")
        print("2 to evaluate a model")
        print("3 to print model summary")
        print("4 to plot distribution of scores in dataset")
        print("5 to plot predicted and true scores")
        print("6 to plot difference error between predicted and true scores")
        print("7 to plot absolute error between predicted and true scores")
        print("0 to exit")

        option = int(input("Enter option: "))
        if option == 0:
            break

        predictor = Predictor()

        match option:
            case 1:
                trainer = PredictorTrainer(predictor)
                trainer.train_model()
            case 2:
                evaluator = PredictorEvaluator(predictor)
                evaluator.evaluate_model()
            case 3:
                predictor.summary()
            case 4:
                plotter = PredictorPlotter()
                plotter.plot_score_distribution()
            case 5:
                plotter = PredictorPlotter()
                plotter.plot_prediction()
            case 6:
                plotter = PredictorPlotter()
                plotter.plot_errors(lambda x, y: x - y, 'Difference')
            case 7:
                plotter = PredictorPlotter()
                plotter.plot_errors(lambda x, y: np.abs(x - y), 'Absolute')

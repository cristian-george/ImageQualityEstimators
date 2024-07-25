import os
import numpy as np

from util.gpu_funcs import check_gpu_support, limit_gpu_memory, increase_cpu_num_threads
from model_config.config_parser import ConfigParser
from model.model_class import ImageQualityPredictor
from model.train_class import PredictorTrainer
from model.evaluate_class import PredictorEvaluator

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    use_gpu = check_gpu_support()
    if use_gpu:
        limit_gpu_memory(memory_limit=3500)
    else:
        increase_cpu_num_threads(num_threads=os.cpu_count())

    while True:
        print("Options: ")
        print("1 to print model summary")
        print("2 to train a model")
        print("3 to evaluate a model")
        print("4 to plot distribution of scores in dataset")
        print("5 to plot predicted and true scores")
        print("6 to plot difference error between predicted and true scores")
        print("7 to plot absolute error between predicted and true scores")
        print("8 to evaluate BRISQUE")
        print("0 to exit")

        option = int(input("Enter option: "))
        if option == 0:
            break

        config_parser = ConfigParser()

        quality_predictor = ImageQualityPredictor(config_parser.get_model_info())
        trainer = PredictorTrainer(config_parser.get_train_info(), quality_predictor)
        evaluator = PredictorEvaluator(config_parser.get_evaluate_info(), quality_predictor)

        match option:
            case 1:
                quality_predictor.summary()
            case 2:
                config_parser.save_train_config()
                trainer.fit_model()
            case 3:
                evaluator.evaluate_model()
            case 4:
                evaluator.plot_score_distribution()
            case 5:
                evaluator.plot_prediction()
            case 6:
                evaluator.plot_errors(lambda x, y: x - y, 'Difference Error vs. True Scores')
            case 7:
                evaluator.plot_errors(lambda x, y: np.abs(x - y), 'Absolute Error vs. True Scores')
            case 8:
                evaluator.evaluate_method('data/LIVE2/LIVE2_matlab_brisque.csv',
                                          method='brisque')

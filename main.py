import os

from model_config.model_config import ConfigParser
from model.model_class import IQA
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
        print("2 to evaluate a model")
        print("0 to exit")

        option = int(input("Enter option: "))
        if option == 0:
            break

        config_parser = ConfigParser('model_config/config.json')

        model = IQA(config_parser.get_model_info(),
                    config_parser.get_learn_info(),
                    config_parser.get_train_info(),
                    config_parser.get_evaluate_info())

        match option:
            case 1:
                config_parser.save_config_json()
                model.train_model()
            case 2:
                model.evaluate_model()

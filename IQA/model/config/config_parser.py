import json


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


class ConfigParser:
    def __init__(self):
        self.model_config_data = load_json('model/config/model_config.json')
        self.train_config_data = load_json('model/config/train_config.json')
        self.evaluate_config_data = load_json('model/config/evaluate_config.json')

    def get_model_info(self):
        return self.model_config_data

    def get_train_info(self):
        return self.train_config_data

    def get_evaluate_info(self):
        return self.evaluate_config_data

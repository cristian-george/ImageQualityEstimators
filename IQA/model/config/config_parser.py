import json
import os


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


class ConfigParser:
    def __init__(self):
        self.model_config_data = load_json('model/config/model_config.json')
        self.train_config_data = load_json('model/config/train_config.json')
        self.evaluate_config_data = load_json('model/config/evaluate_config.json')

    def save_train_config(self):
        ckpt_dir = self.train_config_data.get('callbacks', {}).get('model_checkpoint', {}).get('ckpt_dir', '')

        if ckpt_dir:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            config = {
                "model_config": self.model_config_data,
                "train_config": self.train_config_data,
            }
            with open(os.path.join(ckpt_dir, 'config_model.json'), 'w') as file:
                json.dump(config, file, indent=2)

    def get_model_info(self):
        return self.model_config_data

    def get_train_info(self):
        return self.train_config_data

    def get_evaluate_info(self):
        return self.evaluate_config_data

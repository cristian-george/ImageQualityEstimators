import json
import os


class ConfigParser:
    def __init__(self, config_path='model_config.json'):
        self.file_path = config_path
        self.data = self._load_json()

    def _load_json(self):
        with open(self.file_path, 'r') as file:
            return json.load(file)

    def save_config_json(self):
        ckpt_dir = self.data.get('train_info', {}).get('callbacks', {}).get('best_model_checkpoint', {}).get('ckpt_dir', '')
        if ckpt_dir:
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            config_path = os.path.join(ckpt_dir, 'config.json')
            with open(config_path, 'w') as file:
                json.dump(self.data, file, indent=2)

    def get_model_info(self):
        return self.data.get('model_info', {})

    def get_learn_info(self):
        return self.data.get('learn_info', {})

    def get_train_info(self):
        return self.data.get('train_info', {})

    def get_evaluate_info(self):
        return self.data.get('evaluate_info', {})

import json


class EvaluateConfigParser:
    def __init__(self):
        self.config_path = 'config/evaluate_config.json'
        with open(self.config_path, 'r') as f:
            self._config_data = json.load(f)

    def parse(self):
        data = dict()

        test_dirs = self._config_data.get('test_dirs', '')
        if isinstance(test_dirs, str):
            test_dirs = [test_dirs]
        data['test_dirs'] = test_dirs

        test_lbs = self._config_data.get('test_lbs', None)
        if test_lbs is None:
            data['test_lbs'] = None
        elif isinstance(test_lbs, str):
            data['test_lbs'] = [test_lbs]
        elif isinstance(test_lbs, list):
            data['test_lbs'] = test_lbs
        else:
            raise ValueError("test_lbs must be null, string or list.")

        if data['test_lbs'] is not None and len(data['test_lbs']) != len(data['test_dirs']):
            raise ValueError("test_lbs length must correspond to test_dirs length.")

        data['weights_path'] = self._config_data.get('weights_path', '')
        data['batch_size'] = self._config_data.get('batch_size', 0)

        return data

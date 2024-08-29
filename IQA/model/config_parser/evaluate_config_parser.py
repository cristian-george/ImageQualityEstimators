from model.config_parser.config_parser import ConfigParser


class EvaluateConfigParser(ConfigParser):
    def __init__(self):
        super().__init__('model/config/evaluate_config.json')

    def parse(self):
        data = dict()
        data['root_directory'] = self._config_data.get('root_directory', '')
        data['test_directory'] = self._config_data.get('test_directory', '')
        data['test_lb'] = data['root_directory'] + self._config_data.get('test_lb', '')

        data['weights_path'] = self._config_data.get('weights_path', '')
        data['batch_size'] = self._config_data.get('batch_size')

        return data

from config_parser.config_parser import ConfigParser


class ModelConfigParser(ConfigParser):
    def __init__(self):
        super().__init__('config/model_config.json')

    def parse(self):
        data = dict()
        data['backbone'] = self._config_data.get('backbone', '')
        data['freeze'] = self._config_data.get('freeze')
        data['input_shape'] = tuple(self._config_data.get('input_shape', []))
        data['dense'] = self._config_data.get('dense', [])
        data['dropout'] = self._config_data.get('dropout', [])

        return data

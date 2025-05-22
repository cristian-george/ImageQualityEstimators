from config_parser.config_parser import ConfigParser


class TrainConfigParser(ConfigParser):
    def __init__(self):
        super().__init__('config/train_config.json')

    def parse(self):
        data = dict()
        data['root_directory'] = self._config_data.get('root_directory', '')
        data['train_directory'] = data['root_directory'] + self._config_data.get('train_directory', '')
        data['val_directory'] = data['root_directory'] + self._config_data.get('val_directory', '')
        data['train_lb'] = data['root_directory'] + self._config_data.get('train_lb', '')
        data['val_lb'] = data['root_directory'] + self._config_data.get('val_lb', '')
        data['augment'] = self._config_data.get('augment', False)
        data['batch_size'] = self._config_data.get('batch_size', 0)
        data['epoch_size'] = self._config_data.get('epoch_size', 0)
        data['continue_train'] = self._config_data.get('continue_train', {})
        data['lr'] = self._config_data.get('lr', {})
        data['loss'] = self._config_data.get('loss', {})
        data['callbacks'] = self._config_data.get('callbacks', {})

        return data

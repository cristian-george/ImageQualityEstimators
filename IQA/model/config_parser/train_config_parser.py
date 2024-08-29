from model.config_parser.config_parser import ConfigParser


class TrainConfigParser(ConfigParser):
    def __init__(self):
        super().__init__('model/config/train_config.json')

    def parse(self):
        data = dict()
        data['data_directory'] = self._config_data.get('data_directory', '')
        data['train_directory'] = data['data_directory'] + self._config_data.get('train_directory', '')
        data['val_directory'] = data['data_directory'] + self._config_data.get('val_directory', '')
        data['train_lb'] = data['data_directory'] + self._config_data.get('train_lb', '')
        data['val_lb'] = data['data_directory'] + self._config_data.get('val_lb', '')
        data['augment'] = self._config_data.get('augment')
        data['batch_size'] = self._config_data.get('batch_size')
        data['epoch_size'] = self._config_data.get('epoch_size')
        data['continue_train'] = self._config_data.get('continue_train', {})
        data['lr'] = self._config_data.get('lr', {})
        data['loss'] = self._config_data.get('loss', {})
        data['callbacks'] = self._config_data.get('callbacks', {})

        return data

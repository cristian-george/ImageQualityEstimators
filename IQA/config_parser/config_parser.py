import json
from abc import ABC, abstractmethod


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


class ConfigParser(ABC):
    def __init__(self, config_file):
        self._config_data = load_json(config_file)

    def get_config_data(self):
        return self._config_data

    @abstractmethod
    def parse(self):
        pass

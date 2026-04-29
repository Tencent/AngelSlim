import os
from pathlib import Path
from typing import Any

import yaml


class ConfigManager:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self.load_config()

    def _find_config_file(self) -> Path:
        # 1. 首先检查环境变量
        config_path = os.getenv('PROJECT_CONFIG_PATH')
        if config_path and Path(config_path).exists():
            return Path(config_path)

        # 2. 从当前目录向上查找，直到找到配置文件
        current_dir = Path(__file__).resolve().parent
        while current_dir != current_dir.parent:
            config_file = current_dir.parent / 'config.yaml'
            if config_file.exists():
                return config_file
            current_dir = current_dir.parent

        raise FileNotFoundError(
            'cannot find config.yaml. Please set PROJECT_CONFIG_PATH environment variable or put config.yaml in the project root directory.')

    def load_config(self):
        config_path = self._find_config_file()
        with open(config_path) as f:
            self._config = yaml.safe_load(f)

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    def get_dataset_root(self) -> str:
        """get dataset root path, your dataset file should be in this path"""
        return self._config['DATASETS']['dataset_root']

    def get_dataset_path(self, dataset_name: str) -> str | None:
        """get dataset file path"""
        return self._config['DATASETS']['datasets'].get(dataset_name, None)
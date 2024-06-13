# Copyright (c) 2024 aldcb - GPLv3 (http://gnu.org/licenses/gpl.html)

import os
import yaml

class Config:
    def __init__(self):
        pass

    def update_from_yaml(self, file_path = "configs/config.yml", clear_rest = False):
        """Updates dictionary based on YAML file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file {file_path} doesn't exist")
        if not file_path.lower().endswith((".yaml", ".yml")):
            raise FileNotFoundError(f"Config file {file_path} has unexpected extension")
        if clear_rest:
            self.__dict__.clear()
        with open(file_path, "r") as yaml_file:
            yaml_config = yaml.safe_load(yaml_file)
        self.__dict__.update(yaml_config)

config = Config()

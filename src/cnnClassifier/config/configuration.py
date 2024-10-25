from cnnClassifier.constants import *
import os
import yaml
from cnnClassifier import logger
from pathlib import Path
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig)



class ConfigurationManager:
    def __init__(self, config_path=CONFIG_FILE_PATH):
        self.config = self.read_yaml(config_path)
    
    def read_yaml(self, file_path: Path) -> dict:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)

    def get_data_ingestion_config(self):
        # Extract the data_ingestion config section from the YAML file
        config = self.config["data_ingestion"]
        logger.debug(f"Data Ingestion Config Loaded: {config}")  # Add this debug statement

        # Ensure the directory exists
        create_directories([config["root_dir"]])
    
        # Create and return a DataIngestionConfig object
        data_ingestion_config = DataIngestionConfig(
            root_dir=config["root_dir"],
            kaggle_dataset=config["kaggle_dataset"],
            local_data_file=config["local_data_file"],
            unzip_dir=config["unzip_dir"]
        )

        return data_ingestion_config
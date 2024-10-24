from cnnClassifier.constants import *
import os
from box import ConfigBox
import yaml
from cnnClassifier import logger
from pathlib import Path
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig)



class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = self.read_yaml(config_filepath)
        self.params = self.read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    def read_yaml(self, file_path: Path) -> ConfigBox:
        with open(file_path, "r") as file:
            content = yaml.safe_load(file)
            return ConfigBox(content)  


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
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config
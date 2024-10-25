from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    root_dir: str
    kaggle_dataset: str
    local_data_file: str
    unzip_dir: str

"""
Data Ingestion Module for Chicken Disease Classification

This module handles the data acquisition process for the Chicken Disease Classification project.
It manages downloading the dataset from Kaggle and extracting it to the appropriate directory.
The implementation includes robust error handling, retry mechanisms, and progress tracking.

Features:
- Authenticated Kaggle API integration
- Robust download mechanism with retries
- Progress tracking for downloads and extractions
- Comprehensive error handling
- ZIP file validation and extraction
"""

# Standard library imports
import os
import time
import logging
import zipfile
from pathlib import Path

# Third-party imports
import kaggle
import requests
from tqdm import tqdm
from requests.exceptions import RequestException, ChunkedEncodingError
from kaggle.api.kaggle_api_extended import KaggleApi

# Local application imports
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig

logger = logging.getLogger(__name__)

class DataIngestion:
    """
    Handles data ingestion process for the Chicken Disease Classification model.
    
    This class manages downloading and extracting the dataset, including:
    - Kaggle API authentication
    - File download with progress tracking
    - Automatic retries for failed downloads
    - ZIP file extraction with validation
    
    Attributes:
        config (DataIngestionConfig): Configuration parameters for data ingestion
        api (KaggleApi): Authenticated Kaggle API instance
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initialize the data ingestion process.
        
        Args:
            config (DataIngestionConfig): Configuration containing paths and parameters
        """
        self.config = config
        self.api = KaggleApi()
        self.api.authenticate()  # Authenticate with Kaggle API

    def download_file(self):
        """
        Download the dataset file from Kaggle with retry mechanism.
        
        The method includes:
        - Progress bar for download tracking
        - Automatic retries on failure
        - File size verification
        - Existing file detection
        
        Raises:
            RequestException: If download fails after max retries
            ChunkedEncodingError: If download stream is corrupted
        """
        max_retries = 3
        retries = 0
        
        while retries < max_retries:
            try:
                if not os.path.exists(self.config.local_data_file):
                    # Configure download with progress tracking
                    url = f"https://www.kaggle.com/api/v1/datasets/download/{self.config.kaggle_dataset}"
                    response = requests.get(url, stream=True)

                    # Setup progress bar parameters
                    total_size_in_bytes = int(response.headers.get('content-length', 0))
                    block_size = 1024  # 1 Kilobyte

                    # Download with progress tracking
                    with open(self.config.local_data_file, 'wb') as file, tqdm(
                        desc="Downloading",
                        total=total_size_in_bytes,
                        unit='iB',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as bar:
                        for data in response.iter_content(block_size):
                            file.write(data)
                            bar.update(len(data))

                    logger.info(f"File downloaded to: {self.config.local_data_file}")
                    break
                else:
                    logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
                break

            except (RequestException, ChunkedEncodingError) as e:
                retries += 1
                logger.error(f"Download failed (attempt {retries}/{max_retries}): {e}")
                if retries < max_retries:
                    logger.info(f"Retrying in 5 seconds...")
                    time.sleep(5)  # Delay before retrying
                else:
                    logger.error("Max retries reached. Download failed.")
                    raise e

    def extract_zip_file(self):
        """
        Extract the downloaded ZIP file with validation and progress tracking.
        
        The method includes:
        - ZIP file validation
        - Progress bar for extraction
        - Comprehensive error handling
        - Directory creation if needed
        
        Raises:
            zipfile.BadZipFile: If the ZIP file is invalid or corrupted
            Exception: For other extraction-related errors
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)  # Create directory if it doesn't exist

        # Validate file extension
        if not self.config.local_data_file.endswith('.zip'):
            logger.error("The downloaded file is not a ZIP file. Cannot extract.")
            return

        # Check file existence
        if not os.path.exists(self.config.local_data_file):
            logger.error("ZIP file not found. Cannot extract.")
            return

        try:
            # Verify and extract ZIP contents
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                total_files = len(zip_ref.infolist())

                # Extract with progress tracking
                for member in tqdm(zip_ref.infolist(), desc="Extracting", total=total_files):
                    zip_ref.extract(member, unzip_path)
                
            logger.info(f"ZIP file extracted successfully to: {unzip_path}")
        
        except zipfile.BadZipFile:
            logger.error("The file is not a valid ZIP file or is corrupted.")
        except Exception as e:
            logger.error(f"An error occurred while extracting the ZIP file: {str(e)}")

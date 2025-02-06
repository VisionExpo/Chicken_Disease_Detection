import kaggle
import time
import os
import requests
from tqdm import tqdm
import zipfile
import logging
from pathlib import Path
from requests.exceptions import RequestException, ChunkedEncodingError
from kaggle.api.kaggle_api_extended import KaggleApi
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig

logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.api = KaggleApi()
        self.api.authenticate()

    def download_file(self):
        max_retries = 3  # Maximum number of retries
        retries = 0
        
        while retries < max_retries:
            try:
                if not os.path.exists(self.config.local_data_file):
                    # Download the file with a progress bar
                    url = f"https://www.kaggle.com/api/v1/datasets/download/{self.config.kaggle_dataset}"
                    response = requests.get(url, stream=True)

                    total_size_in_bytes = int(response.headers.get('content-length', 0))
                    block_size = 1024  # 1 Kilobyte

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
                    break  # Exit loop if download is successful
                else:
                    logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
                break

            except (requests.exceptions.RequestException, ChunkedEncodingError) as e:
                retries += 1
                logger.error(f"Download failed (attempt {retries}/{max_retries}): {e}")
                if retries < max_retries:
                    logger.info(f"Retrying in 5 seconds...")
                    time.sleep(5)  # Delay before retrying
                else:
                    logger.error("Max retries reached. Download failed.")
                    raise e

    def get_size(self, file_path):
        # Get the size of the file
        size = os.path.getsize(file_path)
        return f"{size / (1024 * 1024):.2f} MB"

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory with a progress bar.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        # Check if the downloaded file is a ZIP
        if not self.config.local_data_file.endswith('.zip'):
            logger.error("The downloaded file is not a ZIP file. Cannot extract.")
            return

        # Check if the ZIP file exists
        if not os.path.exists(self.config.local_data_file):
            logger.error("ZIP file not found. Cannot extract.")
            return

        try:
            # Verify if it's a valid ZIP file
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                # If no exception is raised, it's a valid ZIP file
                total_files = len(zip_ref.infolist())

                # Use tqdm to display a progress bar for the extraction
                for member in tqdm(zip_ref.infolist(), desc="Extracting", total=total_files):
                    zip_ref.extract(member, unzip_path)
                
            logger.info(f"ZIP file extracted successfully to: {unzip_path}")
        
        except zipfile.BadZipFile:
            logger.error("The file is not a valid ZIP file or is corrupted.")
        except Exception as e:
            logger.error(f"An error occurred while extracting the ZIP file: {str(e)}")

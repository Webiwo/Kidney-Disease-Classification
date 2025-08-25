import os
import zipfile
import gdown
from pathlib import Path
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> Path:
        """Fetch data from Google Drive URL"""

        dataset_url = self.config.source_url
        zip_download_path = Path(self.config.local_data_file)

        os.makedirs(zip_download_path.parent, exist_ok=True)

        logger.info(
            f"Downloading data from {dataset_url} into file {zip_download_path}"
        )

        try:
            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix + file_id, str(zip_download_path), quiet=False)
        except Exception as e:
            logger.error(f"Failed to download file from {dataset_url}: {e}")
            raise

        logger.info(f"Downloaded file size: {get_size(zip_download_path)}")
        return zip_download_path

    def extract_zip_file(self) -> None:
        """Extracts the downloaded ZIP file into the target directory"""

        unzip_path = Path(self.config.unzip_dir)
        os.makedirs(unzip_path, exist_ok=True)

        logger.info(f"Extracting {self.config.local_data_file} into {unzip_path}")

        try:
            with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
                zip_ref.extractall(unzip_path)
        except zipfile.BadZipFile as e:
            logger.error(f"Corrupted zip file {self.config.local_data_file}: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Failed to extract zip file {self.config.local_data_file}: {e}"
            )
            raise

        logger.info(f"Extraction completed successfully: {unzip_path}")

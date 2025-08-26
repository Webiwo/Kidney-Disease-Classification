from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.utils.common import logger

STAGE_NAME = "Data Ingestion stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info(f"*********** {STAGE_NAME} started ***********")
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
        logger.info(f"*********** {STAGE_NAME} completed ***********")
    except Exception as e:
        logger.exception(f"Data ingestion pipeline failed: {e}")
        raise

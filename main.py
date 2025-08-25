from cnnClassifier import logger
from cnnClassifier.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion stage"

if __name__ == "__main__":
    try:
        logger.info(f"*********** {STAGE_NAME} started ***********")
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
        logger.info(f"*********** {STAGE_NAME} completed ***********")
    except Exception as e:
        logger.exception(f"Data ingestion pipeline failed: {e}")
        raise

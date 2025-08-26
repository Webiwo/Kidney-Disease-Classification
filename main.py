from cnnClassifier import logger
from cnnClassifier.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_2_prepare_base_model import (
    PrepareBaseModelTrainingPipeline,
)

STAGE_NAME_1 = "Data Ingestion stage"
STAGE_NAME_2 = "Prepare base model"

if __name__ == "__main__":
    try:
        logger.info(f"*********** {STAGE_NAME_1} started ***********")
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
        logger.info(f"*********** {STAGE_NAME_1} completed ***********")
    except Exception as e:
        logger.exception(f"Data ingestion pipeline failed: {e}")
        raise

    try:
        logger.info(f"*********** {STAGE_NAME_2} started ***********")
        pipeline = PrepareBaseModelTrainingPipeline()
        pipeline.main()
        logger.info(f"*********** {STAGE_NAME_2} completed ***********")
    except Exception as e:
        logger.exception(f"Prepare base model pipeline failed: {e}")
        raise

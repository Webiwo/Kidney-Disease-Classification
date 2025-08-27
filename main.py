from cnnClassifier import logger
from cnnClassifier.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_2_prepare_base_model import (
    PrepareBaseModelTrainingPipeline,
)
from cnnClassifier.pipeline.stage_3_model_training import ModelTrainingPipeline
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.global_flags import DAGSHUB_INITIALIZED
import dagshub


config_manager = ConfigurationManager()
dagshub_config = config_manager.get_dagshub_config()

if dagshub_config.mlflow:
    dagshub.init(
        repo_owner=dagshub_config.repo_owner,
        repo_name=dagshub_config.repo_name,
        mlflow=True,
    )


# STAGE_NAMES = ["Data Ingestion stage", "Prepare base model", "Training model"]

# STAGE_PIPELINES = [
#     DataIngestionTrainingPipeline,
#     PrepareBaseModelTrainingPipeline,
#     ModelTrainingPipeline,
# ]

STAGE_NAMES = ["Prepare base model", "Training model"]

STAGE_PIPELINES = [
    PrepareBaseModelTrainingPipeline,
    ModelTrainingPipeline,
]


if __name__ == "__main__":
    DAGSHUB_INITIALIZED = True
    for stage_name, StagePipeline in zip(STAGE_NAMES, STAGE_PIPELINES):
        try:
            logger.info(f"*********** {stage_name} started ***********")
            logger.info(f"Stage name: {stage_name}")
            if stage_name == "Training model":
                pipeline = StagePipeline(dagshub_config)
            else:
                pipeline = StagePipeline()
            pipeline.main()
            logger.info(f"*********** {stage_name} completed ***********")
        except Exception as e:
            logger.exception(f"{stage_name} pipeline failed: {e}")
            raise

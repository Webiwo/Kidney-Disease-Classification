from cnnClassifier.config.configuration import (
    ConfigurationManager,
    DagshubConfig,
    BaseModelConfig,
)
from cnnClassifier.components.model_training import Training
from cnnClassifier.utils.common import logger
from cnnClassifier.global_flags import DAGSHUB_INITIALIZED
import dagshub

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self, dagshub_config):
        self.dagshub_config = dagshub_config

    def main(self):
        global DAGSHUB_INITIALIZED

        if self.dagshub_config and not DAGSHUB_INITIALIZED:
            dagshub.init(
                repo_owner=self.dagshub_config.repo_owner,
                repo_name=self.dagshub_config.repo_name,
                mlflow=True,
            )
            DAGSHUB_INITIALIZED = True

        config = ConfigurationManager()
        training_config = config.get_training_config()
        base_model_config = config.get_base_model_config()
        training = Training(training_config, base_model_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()


if __name__ == "__main__":
    try:
        logger.info(f"*********** {STAGE_NAME} started ***********")
        dagshub_config = DagshubConfig()
        pipeline = ModelTrainingPipeline(dagshub_config)
        pipeline.main()
        logger.info(f"*********** {STAGE_NAME} completed ***********")
    except Exception as e:
        logger.exception(f"Model training pipeline failed: {e}")
        raise

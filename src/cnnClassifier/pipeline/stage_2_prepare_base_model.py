from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.base_model import BaseModel
from cnnClassifier.utils.common import logger

STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        base_model_config = config_manager.get_base_model_config()
        base_model = BaseModel(config=base_model_config)
        base_model.get_base_model()
        base_model.update_base_model()


if __name__ == "__main__":
    try:
        logger.info(f"*********** {STAGE_NAME} started ***********")
        pipeline = PrepareBaseModelTrainingPipeline()
        pipeline.main()
        logger.info(f"*********** {STAGE_NAME} completed ***********")
    except Exception as e:
        logger.exception(f"Prepare base model pipeline failed: {e}")
        raise

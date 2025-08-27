from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    BaseModelConfig,
    TrainingConfig,
    DagshubConfig,
)
from pathlib import Path


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories(self.config.artifacts_root)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories(config.root_dir)

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_url=config.source_url,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model
        params = self.params

        create_directories(config.root_dir)

        return BaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=tuple(params.IMAGE_SIZE),
            params_learning_rate=params.LEARNING_RATE,
            params_include_top=params.INCLUDE_TOP,
            params_weights=params.WEIGHTS,
            params_classes=params.CLASSES,
        )

    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        base_model = self.config.base_model
        params = self.params

        create_directories(training.root_dir)

        return TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(base_model.updated_base_model_path),
            training_data=Path(self.config.data_ingestion.unzip_dir),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=tuple(params.IMAGE_SIZE),
        )

    def get_dagshub_config(self) -> DagshubConfig:
        dags_hub = self.config.dagshub

        return DagshubConfig(
            repo_owner=dags_hub.repo_owner,
            repo_name=dags_hub.repo_name,
            mlflow=dags_hub.mlflow,
        )

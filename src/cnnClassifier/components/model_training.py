import tensorflow as tf
import mlflow
import mlflow.keras
from cnnClassifier.config.configuration import TrainingConfig, BaseModelConfig
from pathlib import Path


class Training:

    def __init__(
        self, training_config: TrainingConfig, base_model_config: BaseModelConfig
    ):
        self.training_config = training_config
        self.base_model_config = base_model_config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.training_config.updated_base_model_path
        )

    def train_valid_generator(self):
        img_size = self.training_config.params_image_size[:-1]  # (224,224)
        batch_size = self.training_config.params_batch_size

        # --- train dataset ---
        self.train_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.training_config.training_data,
            labels="inferred",
            label_mode="categorical",  # softmax
            batch_size=batch_size,
            image_size=img_size,
            shuffle=True,
            validation_split=0.2,
            subset="training",
            seed=42,
        )

        # --- validation dataset ---
        self.valid_generator = tf.keras.utils.image_dataset_from_directory(
            directory=self.training_config.training_data,
            labels="inferred",
            label_mode="categorical",
            batch_size=batch_size,
            image_size=img_size,
            shuffle=False,
            validation_split=0.2,
            subset="validation",
            seed=42,
        )

        # --- normalization ---
        normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

        self.train_generator = self.train_generator.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        self.valid_generator = self.valid_generator.map(
            lambda x, y: (normalization_layer(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # --- augmentation ---
        if self.training_config.params_is_augmentation:
            data_augmentation = tf.keras.Sequential(
                [
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomRotation(0.1),
                    tf.keras.layers.RandomZoom(0.2),
                    tf.keras.layers.RandomTranslation(0.1, 0.1),
                ]
            )

        self.train_generator = self.train_generator.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # --- pipeline optimalization ---
        self.train_generator = self.train_generator.prefetch(
            buffer_size=tf.data.AUTOTUNE
        )
        self.valid_generator = self.valid_generator.prefetch(
            buffer_size=tf.data.AUTOTUNE
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        with mlflow.start_run():
            mlflow.log_param("epochs", self.training_config.params_epochs)
            mlflow.log_param("batch_size", self.training_config.params_batch_size)
            mlflow.log_param("image_size", self.training_config.params_image_size)
            mlflow.log_param(
                "learning_rate", self.base_model_config.params_learning_rate
            )
            mlflow.log_param(
                "augmentation", self.training_config.params_is_augmentation
            )

        history = self.model.fit(
            self.train_generator,
            epochs=self.training_config.params_epochs,
            validation_data=self.valid_generator,
        )

        for epoch in range(self.training_config.params_epochs):
            mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
            mlflow.log_metric(
                "train_accuracy", history.history["accuracy"][epoch], step=epoch
            )
            mlflow.log_metric(
                "val_loss", history.history["val_loss"][epoch], step=epoch
            )
            mlflow.log_metric(
                "val_accuracy", history.history["val_accuracy"][epoch], step=epoch
            )

        self.save_model(path=self.training_config.trained_model_path, model=self.model)
        mlflow.log_artifact(
            str(self.training_config.trained_model_path), artifact_path="trained_model"
        )

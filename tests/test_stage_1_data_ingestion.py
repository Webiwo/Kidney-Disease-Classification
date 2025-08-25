import pytest
from cnnClassifier.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline


def test_pipeline_runs(monkeypatch):

    called = {"get_config": False, "download_file": False, "extract_zip": False}

    class FakeConfig:
        pass

    class FakeConfigManager:
        def get_data_ingestion_config(self):
            called["get_config"] = True
            return FakeConfig()

    class FakeDataIngestion:
        def __init__(self, config):
            self.config = config
            assert isinstance(config, FakeConfig)

        def download_file(self):
            called["download_file"] = True

        def extract_zip_file(self):
            called["extract_zip"] = True

    monkeypatch.setattr(
        "cnnClassifier.pipeline.stage_1_data_ingestion.ConfigurationManager",
        FakeConfigManager,
    )
    monkeypatch.setattr(
        "cnnClassifier.pipeline.stage_1_data_ingestion.DataIngestion", FakeDataIngestion
    )

    pipeline = DataIngestionTrainingPipeline()
    pipeline.main()

    assert called["get_config"] is True
    assert called["download_file"] is True
    assert called["extract_zip"] is True


def test_pipeline_exception(monkeypatch):

    class FakeConfigManager:
        def get_data_ingestion_config(self):
            raise RuntimeError("Boom!")

    monkeypatch.setattr(
        "cnnClassifier.pipeline.stage_1_data_ingestion.ConfigurationManager",
        FakeConfigManager,
    )

    pipeline = DataIngestionTrainingPipeline()
    with pytest.raises(RuntimeError, match="Boom!"):
        pipeline.main()

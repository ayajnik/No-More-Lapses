from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list

@dataclass(frozen=True)
class DataTransformationConfig:
    data:Path
    root_dir: Path
    transformed_data_path: Path
    trainData: Path
    testData: Path
    training_independent_data: Path
    training_dependent_data: Path
    testing_independent_data: Path
    testing_dependent_data: Path

@dataclass(frozen=True)
class ModelPreparationConfig:
    root_dir: Path
    training_independent_data_path: Path
    training_dependent_data_path: Path
    model_saved_path: Path
    epochs: int
    batch_size: int
    optimizer: str
    loss: str
    metrics: str

@dataclass(frozen=True)
class PredictionPipelineConfig:
    test_independent_data_path: Path
    test_dependent_data_path: Path
    trained_model_path: Path
    predictions_output_path: Path
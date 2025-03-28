from src.No_More_Lapses.constants import *
from src.No_More_Lapses.utils.common import read_yaml, create_directories
from src.No_More_Lapses.entity.config_entity import (DataIngestionConfig,
                                                    DataValidationConfig,
                                                    DataTransformationConfig,
                                                    ModelPreparationConfig,
                                                    PredictionPipelineConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])
        create_directories([config.trainData])
        create_directories([config.testData])

        data_transformation_config = DataTransformationConfig(
            data = config.original_data,
            root_dir=config.root_dir,
            transformed_data_path = config.transformed_data_path,
            trainData = config.trainData,
            testData = config.testData,
            training_independent_data=config.training_independent_data,
            training_dependent_data=config.training_dependent_data,
            testing_independent_data= config.testing_independent_data,
            testing_dependent_data= config.testing_dependent_data
        )

        return data_transformation_config
    
    def get_model_preparation_config(self) -> ModelPreparationConfig:
        config = self.config.model_trainer
        params = self.params.training_hyperparameters

        create_directories([config.root_dir])
        
        data_transformation_config = ModelPreparationConfig(
            root_dir=config.root_dir,
            training_independent_data_path = config.training_independent_data_path,
            training_dependent_data_path = config.training_dependent_data_path,
            model_saved_path = config.model_saved_path,
            epochs=params.EPOCHS,
            batch_size=params.BATCH_SIZE,
            optimizer=params.OPTIMIZER,
            loss=params.LOSS,
            metrics=params.METRICS
        )

        return data_transformation_config
    
    def get_prediction_pipeline_config(self) -> PredictionPipelineConfig:
        config = self.config.prediction_pipeline

        
        data_transformation_config = PredictionPipelineConfig(
            test_independent_data_path = config.test_independent_data_path,
            test_dependent_data_path = config.test_dependent_data_path,
            trained_model_path = config.trained_model_path,
            predictions_output_path=config.predictions_output_path
        )

        return data_transformation_config
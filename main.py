from src.No_More_Lapses import logger
from src.No_More_Lapses.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.No_More_Lapses.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.No_More_Lapses.pipeline.stage_03_data_transformer import DataTransformerTrainingPipeline
from src.No_More_Lapses.pipeline.stage_04_base_model import BaselineModelTrainingPipeline
from src.No_More_Lapses.pipeline.stage_05_attention_model import AttentionModelTrainingPipeline
from src.No_More_Lapses.pipeline.stage_06_prediction import PredictionPipelineStage


STAGE_NAME = "Data Ingestion stage"

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Validation stage"

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Transformation stage"

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformerTrainingPipeline()
   data_transformation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Baseline Model stage"

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   baselining = BaselineModelTrainingPipeline()
   baselining.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Attention Model stage"

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   baselining = AttentionModelTrainingPipeline()
   baselining.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Prediction stage"

try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   baselining = PredictionPipelineStage()
   baselining.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
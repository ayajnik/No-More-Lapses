from src.No_More_Lapses.config.configuration import ConfigurationManager
from src.No_More_Lapses.components.prediction import PredictionPipeline
from src.No_More_Lapses import logger

class PredictionPipelineStage:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_prediction_config = config.get_prediction_pipeline_config()
        predictive_pipeline = PredictionPipeline(config=data_prediction_config)
        predictive_pipeline.run_prediction()

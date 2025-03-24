from src.No_More_Lapses.config.configuration import ConfigurationManager
from src.No_More_Lapses.components.baseline_model import RandomForestTrainer
from src.No_More_Lapses import logger

class BaselineModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformer_config = config.get_model_preparation_config()
        baseline_modeling = RandomForestTrainer(config=data_transformer_config)
        baseline_modeling.train_and_log()
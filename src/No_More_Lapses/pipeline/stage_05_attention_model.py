from src.No_More_Lapses.config.configuration import ConfigurationManager
from src.No_More_Lapses.components.attention_model import ModelPreparation
from src.No_More_Lapses import logger

class AttentionModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformer_config = config.get_model_preparation_config()
        attention_modeling = ModelPreparation(config=data_transformer_config)
        attention_modeling.trainModel()

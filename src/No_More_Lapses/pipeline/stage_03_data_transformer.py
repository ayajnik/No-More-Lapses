from src.No_More_Lapses.config.configuration import ConfigurationManager
from src.No_More_Lapses.components.data_transformer import DataTransformer
from src.No_More_Lapses import logger


class DataTransformerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformer_config = config.get_data_transformation_config()
        data_validation = DataTransformer(config=data_transformer_config)
        data_validation.train_test_split()
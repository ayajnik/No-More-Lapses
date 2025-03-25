import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
import os
from sklearn.preprocessing import StandardScaler
from src.No_More_Lapses import logger
from src.No_More_Lapses.components.attention_model import AttentionLayer
from src.No_More_Lapses.entity.config_entity import PredictionPipelineConfig

class PredictionPipeline:
    def __init__(self, config: PredictionPipelineConfig):
        self.config = config

    def run_prediction(self):
        # Load test data
        X_test = pd.read_csv(self.config.test_independent_data_path)
        y_test = pd.read_csv(self.config.test_dependent_data_path)

        # Drop index column if present
        X_test = X_test.drop(columns=['Unnamed: 0'], errors='ignore')
        y_test = y_test.drop(columns=['Unnamed: 0'], errors='ignore')

        logger.info("Loaded and cleaned test data.")

        # Scale test data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_test)  # ⚠️ If you saved the training scaler, load and use it here instead

        # Load trained model

        model = tf.keras.models.load_model(
                self.config.trained_model_path,
                compile=False,
                custom_objects={"AttentionLayer": AttentionLayer}
            )

        logger.info("Trained model loaded.")

        # Predict probabilities
        y_pred_probs = model.predict(X_scaled)
        y_pred_labels = np.argmax(y_pred_probs, axis=1)

        # Add lapse_calculator column to test DataFrame
        results = X_test.copy()
        results["lapse_calculator"] = y_pred_labels

        # Save result
        os.makedirs(os.path.dirname(self.config.predictions_output_path), exist_ok=True)
        results.to_csv(self.config.predictions_output_path, index=False)
        logger.info(f"Predictions saved to: {self.config.predictions_output_path}")

        # Log as MLflow artifact
        with mlflow.start_run(run_name="attentionModel_prediction"):
            mlflow.log_artifact(self.config.predictions_output_path)

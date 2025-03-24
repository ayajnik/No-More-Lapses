from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import mlflow
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.No_More_Lapses import logger
from src.No_More_Lapses.entity.config_entity import ModelPreparationConfig

class RandomForestTrainer:
    def __init__(self, model_save_path="artifacts/model_trainer/baseline_rf_model.pkl",config= ModelPreparationConfig):
        self.config = config
        self.model_path = model_save_path
        self.model = RandomForestClassifier()

    def train_and_log(self):


        X_train = pd.read_csv(self.config.training_independent_data_path)
        y_train = pd.read_csv(self.config.training_dependent_data_path)
        logger.info("Train data loaded")

        X_train = X_train.drop(columns=['Unnamed: 0'],errors='ignore')
        y_train = y_train.drop(columns=['Unnamed: 0'],errors='ignore')
        logger.info("Removing the index column for the preparation of baseline model.")

        y_series = y_train["POLICY STATUS"]  # explicitly get the column
        num_classes = len(y_series.unique())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        logger.info("Applied Standard Scaler to scale down the values in the same range to avoid gradient boosting or biases for the baseline model.")

        clf = RandomForestClassifier()
        clf.fit(X_scaled, y_series)
        y_pred = clf.predict(X_scaled)
        acc = accuracy_score(y_series, y_pred)
        report=classification_report(y_series, y_pred)
        joblib.dump(clf, 'artifacts/model_trainer/baseline_model.h5')

        with open("logs/rf_report.txt", "w") as f:
            f.write(report)

        with open("artifacts/model_trainer/baseline_rf_report.txt", "w") as f:
            f.write(report)

        with mlflow.start_run(run_name="baseline_RandomForestClassifier"):
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(clf, "model", registered_model_name="baseline_RandomForestClassifier")
            mlflow.log_artifact('artifacts/model_trainer')
            mlflow.log_artifact("artifacts/model_trainer/baseline_rf_report.txt")
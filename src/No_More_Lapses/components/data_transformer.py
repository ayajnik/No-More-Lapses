from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from src.No_More_Lapses import logger
from src.No_More_Lapses.entity.config_entity import DataTransformationConfig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class DataTransformer:
    
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def encode_object_columns(self):

        global df_non_categorical, object_columns
        
        # 1. Select columns with object dtype
        logger.info("Reading the original dataset for extracting object data type")
        dataframe = pd.read_csv(self.config.data, on_bad_lines='skip')
        logger.info("Extracting columns with object data type")
        object_columns = dataframe.select_dtypes(include=['object']).columns
        logger.info("creating a dataframe with columns with no object data type columns")
        df_non_categorical = dataframe.drop(columns=object_columns)

        logger.info("Creating Ordinal encoder object")
        encoder = OrdinalEncoder()
        df_encoded = pd.DataFrame(encoder.fit_transform(dataframe[object_columns]), 
                                columns=object_columns, 
                                index=dataframe.index)
        logger.info("Encoded the object columns and created a dataframe from it.")
        
        return df_encoded
    
    def extract_target(self):

        encoded_df = self.encode_object_columns()
        target = encoded_df['POLICY STATUS']
        logger.info("Creating a variable with encoded target variable.")
        return target
    
    def apply_chi_squared_test(self):

        global dependent_var
        encoded_data = self.encode_object_columns()
        dependent_var = self.extract_target()

        selector = SelectKBest(chi2, k='all')  # Select all features initially
        selector.fit(encoded_data, dependent_var)
        logger.info("Applied Chi-Squared test to get the most important features")

        # Get feature scores and p-values
        scores = pd.DataFrame({
            'feature': object_columns,
            'score': selector.scores_,
            'p_value': selector.pvalues_
        })
        top_features = scores.head(5)['feature'].tolist()

        # Sort features by score in descending order
        scores = scores.sort_values('score', ascending=False)
        scores.to_csv(self.config.root_dir+'/scores.csv')
        logger.info("Exported Chi-squared test to data transformation root directory.")

        df_selected = encoded_data[top_features]
        logger.info("Created a separate dataframe with encoded important features")

        return df_selected
    
    def final_dataframe(self):

        cat_encoded_df = self.apply_chi_squared_test()

        logger.info("Concatenating the encoded categorical columns with numerical/float columns")

        df_final = pd.concat([df_non_categorical, cat_encoded_df], axis=1)
        df_final.to_csv(self.config.transformed_data_path)
        logger.info("Exported the encded data for archival purposes.")
        return df_final
    
    def train_test_split(self):
        
        imputer = SimpleImputer(strategy='median')

        final_encoded_dataframe = self.final_dataframe()
        # Drop 'POLICY STATUS' and 'Unnamed: 0' from the features
        X = final_encoded_dataframe.drop(['POLICY STATUS', 'Unnamed: 0'], axis=1, errors='ignore')

        X_train, X_test, y_train, y_test = train_test_split(X, dependent_var, test_size=0.2, random_state=42)
        logger.info("Data has been bifurcated into training and testing dataset")

        X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        logger.info("SimpleImputer has been applied to the training data")


        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)
        logger.info("SMOTE has been applied for class imabalnce problem.")

        X_train_resampled.to_csv(self.config.training_independent_data)
        y_train_resampled.to_csv(self.config.training_dependent_data)

        logger.info("Training encoded final data exported.")

        X_test.to_csv(self.config.testing_independent_data, index=False)
        y_test.to_csv(self.config.testing_dependent_data, index=False)
        logger.info("Exported the test dataset")


        return X_train_resampled, y_train_resampled, X_test, y_test
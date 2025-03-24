import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.No_More_Lapses import logger
from src.No_More_Lapses.entity.config_entity import ModelPreparationConfig
import mlflow

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weights",
                                 shape=(input_shape[-1], 1),
                                 initializer="glorot_uniform",
                                 trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        # Compute attention scores
        e = tf.matmul(inputs, self.W)
        alpha = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(alpha * inputs, axis=1)
        return context, tf.squeeze(alpha, -1)  # output attention weights too

def build_attention_model(input_dim, num_classes, return_attention=False):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.expand_dims(inputs, axis=1)

    attention_layer = AttentionLayer()
    attention_output, attention_weights = attention_layer(x)

    x = tf.keras.layers.Dense(64, activation='relu')(attention_output)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    if return_attention:
        model = tf.keras.Model(inputs=inputs, outputs=[output, attention_weights])
    else:
        model = tf.keras.Model(inputs=inputs, outputs=output)

    return model

class ModelPreparation:
    
    def __init__(self, config: ModelPreparationConfig):
        self.config = config

    def trainModel(self):
        X_train = pd.read_csv(self.config.training_independent_data_path)
        y_train = pd.read_csv(self.config.training_dependent_data_path)
        logger.info("Train data loaded")

        X_train = X_train.drop(columns=['Unnamed: 0'],errors='ignore')
        y_train = y_train.drop(columns=['Unnamed: 0'],errors='ignore')
        logger.info("Removing the index column")

        y_series = y_train["POLICY STATUS"]  # explicitly get the column
        num_classes = len(y_series.unique())

        # Step 4: Convert to categorical
        y_encoded = tf.keras.utils.to_categorical(y_series, num_classes)
        logger.info("Converted y to one-hot encoding")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        logger.info("Applied Standard Scaler to scale down the values in the same range to avoid gradient boosting or biases")

        model = build_attention_model(input_dim=X_scaled.shape[1], num_classes=num_classes, return_attention=False)


        model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
            )
        
        model.fit(X_scaled, y_encoded, epochs=self.config.epochs, batch_size=self.config.batch_size)
        attention_model = build_attention_model(input_dim=X_scaled.shape[1], num_classes=num_classes, return_attention=True)
        attention_model.set_weights(model.get_weights())
        loss, acc = model.evaluate(X_scaled, y_encoded)
        # Save model and scaler
        model.save(self.config.model_saved_path)
        
        with mlflow.start_run(run_name="model_attentionModel"):
            mlflow.log_param("model_type", "attention_nn")
            mlflow.log_param("epochs", self.config.epochs)
            mlflow.log_param("batch_size", self.config.batch_size)
            mlflow.log_metric("accuracy", acc)
            mlflow.tensorflow.log_model(model, "model", registered_model_name="model_attentionModel")
            mlflow.log_artifact(self.config.root_dir)




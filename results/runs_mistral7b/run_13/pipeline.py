import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, pyfunc, traceback

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Preprocessing steps like filling NaN, encoding categorical variables, scaling features, etc.
    # For simplicity, I'll just drop any rows with NaN values
    data.dropna(inplace=True)
    X = data.drop('Anomaly', axis=1)
    y = data['Anomaly']
    return X, y

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    mlflow.pyfunc.log_model(
        pyfunc.DistributedModel.from_flask(model, "RandomForestAnomalyDetector"),
        artifact_path="model"
    )

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_param, log_artifact, start_run, end_run, log_metric

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Perform any necessary preprocessing (e.g., normalization, encoding categorical variables)
    # For simplicity, I'll assume no preprocessing is needed
    return data, data['Anomaly']

def train_model(X_train, y_train):
    with start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        log_artifact(model, "model")

        # Log some metrics for tracking
        log_metric("train_score", model.score(X_train, y_train))
        log_metric("test_score", model.score(X_test, y_test))

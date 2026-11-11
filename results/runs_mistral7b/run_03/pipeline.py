import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow import log_artifact, track, pyfunc, run_local

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Any preprocessing steps like normalization, encoding, etc.
    # For simplicity, I'll just return the data as is.
    return data.drop('Anomaly', axis=1), data['Anomaly']

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.pyfunc.log_model(model, "model")
        track(model.score(X_train, y_train))

    return model

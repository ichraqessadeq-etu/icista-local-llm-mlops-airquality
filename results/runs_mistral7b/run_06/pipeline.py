import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data):
    # Add any necessary preprocessing steps like normalization, encoding, etc.
    # Return X and y as separate dataframes

def train_model(X_train, y_train):
    with mlflow.start_run():
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        mlflow.log_artifact(f"model.pkl", local_path="model.pkl")
        mlflow.log_param("num_estimators", model.n_estimators)
        mlflow.log_param("max_depth", model.max_depth)

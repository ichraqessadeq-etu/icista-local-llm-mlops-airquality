import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow import log_param, log_metric, log_model

def load_data(file_path):
    """Load data from the specified file path."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Failed to load data: {e}")

def preprocess_data(data):
    """Preprocess the data by separating features and target."""
    try:
        X = data.drop(columns=['Anomaly'])
        y = data['Anomaly']
        return X, y
    except Exception as e:
        raise Exception(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log the model using MLflow."""
    try:
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Log model parameters and metrics
        log_param("n_estimators", model.n_estimators)
        log_param("max_depth", model.max_depth)
        log_metric("accuracy", accuracy_score(y_train, model.predict(X_train)))
        
        # Log the model
        log_model(model, "random_forest_model")
        
        return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")

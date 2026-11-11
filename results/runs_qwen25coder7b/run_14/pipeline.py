import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mlflow import log_param, log_metric, log_model

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Failed to load data: {e}")

def preprocess_data(data):
    """Preprocess data to separate features and target."""
    try:
        X = data.drop(columns=['Anomaly'])
        y = data['Anomaly']
        return X, y
    except Exception as e:
        raise Exception(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """Train a RandomForestClassifier and log metrics."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model parameters
        log_param("n_estimators", 100)
        log_param("random_state", 42)
        
        # Log model performance
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        log_metric("accuracy", accuracy)
        
        # Log the model
        log_model(model, "random_forest_model")
        
        return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")

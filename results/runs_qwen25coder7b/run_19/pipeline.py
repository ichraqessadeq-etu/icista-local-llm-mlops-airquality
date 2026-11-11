import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: The loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise Exception(f"Failed to load data: {e}")

def preprocess_data(data):
    """
    Preprocess the data by separating features and target.
    
    Args:
    data (pd.DataFrame): The input data.
    
    Returns:
    tuple: A tuple containing the features (X) and target (y).
    """
    try:
        X = data.drop(columns=['Anomaly'])
        y = data['Anomaly']
        return X, y
    except Exception as e:
        raise Exception(f"Failed to preprocess data: {e}")

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier model.
    
    Args:
    X_train (pd.DataFrame): The training features.
    y_train (pd.Series): The training target.
    
    Returns:
    RandomForestClassifier: The trained model.
    """
    try:
        with mlflow.start_run():
            mlflow.log_param("model", "RandomForestClassifier")
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_metric("accuracy", accuracy_score(y_train, model.predict(X_train)))
            return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")

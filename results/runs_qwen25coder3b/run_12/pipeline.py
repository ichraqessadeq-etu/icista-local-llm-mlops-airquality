import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.models import save_model

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Drop rows with missing values
        data = data.dropna()
        
        # Define features and target
        X = data.drop('Anomaly', axis=1)
        y = data['Anomaly']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise ValueError(f"Error preprocessing data: {e}")

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise ValueError(f"Error training model: {e}")

def save_model(model, file_path):
    """Save the model using MLflow."""
    try:
        mlflow.set_experiment("AirQualityAnomalyDetection")
        with mlflow.start_run():
            mlflow.log_params({
                "n_estimators": 100,
                "random_state": 42
            })
            
            # Log the model
            signature = infer_signature(X_train, model.predict(X_train))
            save_model(model, file_path, signature=signature)
            print(f"Model saved at {file_path}")
    except Exception as e:
        raise ValueError(f"Error saving model: {e}")

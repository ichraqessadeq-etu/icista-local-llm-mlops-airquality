import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load the dataset from the specified file path."""
    try:
        logging.info("Loading data from %s", file_path)
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except pd.errors.EmptyDataError:
        logging.error("Empty data in file: %s", file_path)
        raise

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target, and handling missing values."""
    try:
        logging.info("Preprocessing data")
        X = data.drop(columns=['Anomaly'])
        y = data['Anomaly']
        X = X.fillna(X.mean())  # Fill missing values with mean
        return X, y
    except Exception as e:
        logging.error("Error preprocessing data: %s", str(e))
        raise

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    try:
        logging.info("Training model")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        logging.error("Error training model: %s", str(e))
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    try:
        logging.info("Evaluating model")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info("Accuracy: %.2f", accuracy)
        return accuracy
    except Exception as e:
        logging.error("Error evaluating model: %s", str(e))
        raise

def main():
    try:
        # Load data
        data = load_data('data/air_quality.csv')
        
        # Preprocess data
        X, y = preprocess_data(data)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test)
        
        # Log model to MLflow
        mlflow.set_experiment("air_quality_anomaly_detection")
        with mlflow.start_run():
            mlflow.log_param("n_estimators", 100)
            mlflow.log_metric("accuracy", evaluate_model(model, X_test, y_test))
            mlflow.sklearn.log_model(model, "anomaly_detector")
    except Exception as e:
        logging.error("Error in pipeline: %s", str(e))

if __name__ == "__main__":
    main()

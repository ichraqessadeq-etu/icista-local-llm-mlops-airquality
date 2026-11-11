import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from the specified CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError("File not found.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the CSV file.")

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target."""
    try:
        X = data.drop('Anomaly', axis=1)
        y = data['Anomaly']
        return X, y
    except KeyError:
        raise ValueError("Target column 'Anomaly' not found in the data.")

def train_model(X_train, y_train):
    """Train a RandomForest model and log it using MLflow."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, 'random_forest_model')
            return model
    except Exception as e:
        mlflow.log_exception(e)
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and log the accuracy."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric('accuracy', accuracy)
        return accuracy
    except Exception as e:
        mlflow.log_exception(e)
        raise

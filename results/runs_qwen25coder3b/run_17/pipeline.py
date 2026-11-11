import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError("File not found")
    except pd.errors.EmptyDataError:
        raise ValueError("Empty data in file")

def preprocess_data(data):
    """Preprocess the data by splitting into features and target."""
    try:
        X = data.drop('Anomaly', axis=1)
        y = data['Anomaly']
        return X, y
    except KeyError:
        raise ValueError("Target column 'Anomaly' not found")

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise ValueError(f"Error training model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's accuracy."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    except Exception as e:
        raise ValueError(f"Error evaluating model: {e}")

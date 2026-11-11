import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def load_data(file_path):
    """Load the dataset from the specified file path."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise ValueError("The file path does not exist.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")

def preprocess_data(data):
    """Preprocess the data by splitting into features and target, and handling missing values."""
    try:
        X = data.drop(columns=['Anomaly'])
        y = data['Anomaly']
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        raise ValueError(f"An error occurred during preprocessing: {e}")

def train_model(X_train, y_train):
    """Train a RandomForest classifier and log the model to MLflow."""
    try:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            mlflow.sklearn.log_model(model, "model")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            return model
    except Exception as e:
        raise ValueError(f"An error occurred during model training: {e}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from sklearn.exceptions import NotFittedError

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess data by separating features and target."""
    X = data.drop(columns=['Anomaly'])
    y = data['Anomaly']
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        mlflow.log_param("n_estimators", 100)
        mlflow.sklearn.log_model(model, "model")
        return model

def predict(model, X):
    """Make predictions using the trained model."""
    try:
        return model.predict(X)
    except NotFittedError:
        raise Exception("Model is not fitted. Please train the model first.")

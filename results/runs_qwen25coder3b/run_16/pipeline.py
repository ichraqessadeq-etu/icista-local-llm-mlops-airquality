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
        raise ValueError(f"File not found: {file_path}")

def preprocess_data(data):
    """Preprocess the data by splitting features and target."""
    features = data.drop('Anomaly', axis=1)
    target = data['Anomaly']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train a RandomForestClassifier model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def log_model(model, experiment_name, run_name):
    """Log the model using MLflow."""
    with mlflow.start_run(experiment_name=experiment_name, run_name=run_name):
        mlflow.sklearn.log_model(model, 'model')
        mlflow.log_param('n_estimators', model.n_estimators)
        mlflow.log_metric('accuracy', accuracy_score(y_train, model.predict(X_train)))

if __name__ == "__main__":
    file_path = 'data/air_quality.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    log_model(model, experiment_name='AirQualityAnomalyDetection', run_name='Run1')

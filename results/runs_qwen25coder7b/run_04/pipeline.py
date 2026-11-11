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
    """Train a RandomForestClassifier model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model parameters and metrics
        log_param("n_estimators", 100)
        log_metric("accuracy", model.score(X_train, y_train))
        
        return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")

def main():
    """Main function to load data, preprocess, and train model."""
    try:
        data = load_data('data/air_quality.csv')
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        
        # Log model
        log_model(model, artifact_path="model")
    except Exception as e:
        raise Exception(f"Failed to execute main pipeline: {e}")

if __name__ == "__main__":
    main()

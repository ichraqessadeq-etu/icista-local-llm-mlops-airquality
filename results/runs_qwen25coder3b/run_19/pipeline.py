import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("Error: File not found.")
        return None

def preprocess_data(data):
    """Preprocess the data by splitting into features and target."""
    try:
        X = data.drop('Anomaly', axis=1)
        y = data['Anomaly']
        return X, y
    except KeyError:
        print("Error: Target column 'Anomaly' not found.")
        return None

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

def main():
    file_path = 'data/air_quality.csv'
    data = load_data(file_path)
    if data is not None:
        X, y = preprocess_data(data)
        if X is not None and y is not None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = train_model(X_train, y_train)
            if model is not None:
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Model accuracy: {accuracy}")
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_artifact("model.pkl", artifact_path="model")

if __name__ == "__main__":
    main()

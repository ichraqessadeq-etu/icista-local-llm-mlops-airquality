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
        raise FileNotFoundError(f"File {file_path} not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"File {file_path} is empty.")

def preprocess_data(data):
    """Preprocess the data by splitting it into features and target."""
    try:
        features = data.drop('Anomaly', axis=1)
        target = data['Anomaly']
        return features, target
    except KeyError:
        raise KeyError("Dataframe does not contain 'Anomaly' column.")

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise Exception(f"Error training model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's accuracy."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy
    except Exception as e:
        raise Exception(f"Error evaluating model: {e}")

def run_pipeline(file_path):
    """Run the entire pipeline."""
    try:
        data = load_data(file_path)
        features, target = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")
        with mlflow.start_run():
            mlflow.log_param("model", "RandomForest")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.sklearn.log_model(model, "model")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_pipeline('data/air_quality.csv')

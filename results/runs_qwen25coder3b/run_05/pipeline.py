import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        raise MlflowException("File not found: {}".format(file_path))
    except pd.errors.EmptyDataError:
        raise MlflowException("Empty data in the file: {}".format(file_path))

def preprocess_data(data):
    """Preprocess the data."""
    # Drop rows with missing values
    data = data.dropna()
    
    # Convert target column to binary
    data['Anomaly'] = (data['Anomaly'] > 0).astype(int)
    
    # Split features and target
    X = data.drop('Anomaly', axis=1)
    y = data['Anomaly']
    
    return X, y

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        raise MlflowException("Error training the model: {}".format(str(e)))

def run_pipeline(file_path):
    """Run the entire pipeline."""
    try:
        data = load_data(file_path)
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with mlflow.start_run():
            model = train_model(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            
            mlflow.sklearn.log_model(model, "model")
            
            print("Model trained with accuracy: {:.2f}".format(accuracy))
    except MlflowException as e:
        print("MLflow exception: {}".format(e))

if __name__ == "__main__":
    run_pipeline("data/air_quality.csv")

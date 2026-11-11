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
        raise ValueError("File not found at the specified path.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or contains no data.")

def preprocess_data(data):
    """Preprocess the data for training."""
    try:
        # Define features and target
        features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'Hour', 'DayOfWeek', 'Month']
        target = 'Anomaly'
        
        # Split the data into features and target
        X = data[features]
        y = data[target]
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except KeyError:
        raise ValueError("One or more required columns are missing from the data.")

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    try:
        # Initialize the model
        model = RandomForestClassifier(random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(model, "anomaly_detector")
        
        return model
    except Exception as e:
        raise ValueError(f"An error occurred during model training: {e}")

def predict_anomalies(model, X_test):
    """Predict anomalies using the trained model."""
    try:
        # Predict anomalies
        predictions = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model accuracy: {accuracy:.2f}")
        
        return predictions
    except Exception as e:
        raise ValueError(f"An error occurred during prediction: {e}")

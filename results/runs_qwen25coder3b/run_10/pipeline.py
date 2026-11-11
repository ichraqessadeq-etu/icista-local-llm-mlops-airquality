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
    except pd.errors.ParserError:
        raise ValueError(f"Failed to parse CSV file {file_path}.")

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Select features and target column
        features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'Hour', 'DayOfWeek', 'Month']
        target = 'Anomaly'
        
        # Split data into features and target
        X = data[features]
        y = data[target]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except KeyError:
        raise KeyError(f"Missing column(s) in data: {features}")

def train_model(X_train, y_train):
    """Train a RandomForest classifier."""
    try:
        # Initialize the RandomForest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(model, "anomaly_detector")
        
        return model
    except Exception as e:
        raise Exception(f"Failed to train model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.2f}")
        
        return accuracy
    except Exception as e:
        raise Exception(f"Failed to evaluate model: {e}")

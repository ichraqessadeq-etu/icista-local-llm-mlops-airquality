import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_data(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logger.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logger.error("File not found.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading data: {e}")
        raise

def preprocess_data(data):
    """Preprocess the data."""
    try:
        # Define features and target
        features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'Hour', 'DayOfWeek', 'Month']
        target = 'Anomaly'
        
        # Split the data
        X = data[features]
        y = data[target]
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info("Data preprocessed successfully.")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        logger.error(f"An error occurred while preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    """Train the RandomForest model."""
    try:
        # Initialize the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        logger.info("Model trained successfully.")
        return model
    except Exception as e:
        logger.error(f"An error occurred while training the model: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model evaluation: Accuracy = {accuracy:.2f}")
        return accuracy
    except Exception as e:
        logger.error(f"An error occurred while evaluating the model: {e}")
        raise

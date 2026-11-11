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
        raise FileNotFoundError("The file does not exist.")
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError("The file is empty.")

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
        
        return X_train, X_test, y_train, y_test
    except KeyError:
        raise KeyError("The target column or feature columns are not in the dataset.")

def train_model(X_train, y_train):
    """Train a RandomForest model."""
    try:
        # Initialize the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        return model
    except Exception as e:
        raise Exception(f"An error occurred during model training: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model."""
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    except Exception as e:
        raise Exception(f"An error occurred during model evaluation: {e}")

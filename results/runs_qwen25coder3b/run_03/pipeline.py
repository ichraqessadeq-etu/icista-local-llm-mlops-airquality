import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        raise MlflowException(f"Error loading data: {e}")

def preprocess_data(data):
    try:
        # Example preprocessing: drop columns with high correlation
        data = data.drop(['PT08.S1(CO)', 'PT08.S2(NMHC)', 'T', 'RH', 'AH'], axis=1)
        
        # Encode categorical variables
        data = pd.get_dummies(data, columns=['DayOfWeek', 'Month'])
        
        # Split features and target
        X = data.drop('Anomaly', axis=1)
        y = data['Anomaly']
        
        return X, y
    except Exception as e:
        raise MlflowException(f"Error preprocessing data: {e}")

def train_model(X_train, y_train):
    try:
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "anomaly_detector")
        
        return model
    except Exception as e:
        raise MlflowException(f"Error training model: {e}")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
import logging

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def preprocess_data(data):
    try:
        # Select relevant features
        X = data[['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'NOx(GT)', 'C6H6(GT)', 'T', 'RH', 'AH', 'Hour', 'DayOfWeek', 'Month']]
        
        # Encode categorical variables
        X['Hour'] = pd.Categorical(X['Hour']).codes
        X['DayOfWeek'] = pd.Categorical(X['DayOfWeek']).codes
        X['Month'] = pd.Categorical(X['Month']).codes
        
        # Scale data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, data['Anomaly'], test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def train_model(X_train, y_train):
    try:
        # Train RandomForest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Log model with MLflow
        mlflow.sklearn.log_model(model, "random_forest")
        
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

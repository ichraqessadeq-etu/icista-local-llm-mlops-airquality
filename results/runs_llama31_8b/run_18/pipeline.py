import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow.sklearn

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    try:
        # Drop unnecessary columns
        data.drop(['Hour', 'DayOfWeek', 'Month'], axis=1, inplace=True)
        
        # Convert categorical variables to numerical
        data['AH'] = pd.Categorical(data['AH']).codes
        
        # Scale data
        scaler = StandardScaler()
        X = scaler.fit_transform(data.drop('Anomaly', axis=1))
        y = data['Anomaly']
        return X, y
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None, None

def train_model(X_train, y_train):
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        with mlflow.start_run():
            mlflow.sklearn.log_model(model, "random_forest")
            model.fit(X_train, y_train)
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

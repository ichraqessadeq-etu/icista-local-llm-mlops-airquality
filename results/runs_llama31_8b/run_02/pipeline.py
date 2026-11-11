import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow
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
        # Handle missing values
        data.fillna(data.mean(), inplace=True)

        # One-hot encoding for categorical features
        data = pd.get_dummies(data, columns=['Hour', 'DayOfWeek', 'Month'])

        # Split data into features (X) and target (y)
        X = data.drop(['Anomaly', 'PT08.S1(CO)_mean', 'PT08.S2(NMHC)_mean', 'PT08.S3(NOx)_mean', 'PT08.S4(NO2)_mean', 'PT08.S5(O3)_mean', 'NOx(GT)_mean', 'C6H6(GT)_mean', 'T_mean', 'RH_mean', 'AH_mean'], axis=1)
        y = data['Anomaly']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features using StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None

def train_model(X_train, y_train):
    try:
        # Train RandomForestClassifier model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")

        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

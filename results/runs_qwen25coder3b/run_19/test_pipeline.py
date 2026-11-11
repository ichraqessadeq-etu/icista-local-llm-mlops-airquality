import pytest
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    file_path = 'data/air_quality.csv'
    data = load_data(file_path)
    assert data is not None

def test_preprocess_data():
    file_path = 'data/air_quality.csv'
    data = load_data(file_path)
    X, y = preprocess_data(data)
    assert X is not None and y is not None

def test_train_model():
    file_path = 'data/air_quality.csv'
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert model is not None

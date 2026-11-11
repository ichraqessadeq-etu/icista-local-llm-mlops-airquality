import pytest
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    data = load_data('data/air_quality.csv')
    assert data is not None

def test_preprocess_data():
    data = load_data('data/air_quality.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert X_train is not None and X_test is not None and y_train is not None and y_test is not None

def test_train_model():
    data = load_data('data/air_quality.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert model is not None

import pytest
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    data = load_data('data/air_quality.csv')
    assert data is not None

def test_preprocess_data():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    assert X is not None
    assert y is not None

def test_train_model():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert model is not None

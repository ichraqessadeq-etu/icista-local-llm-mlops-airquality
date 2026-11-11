import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    data = load_data('data/air_quality.csv')
    assert not data.empty
    assert 'Anomaly' in data.columns

def test_preprocess_data():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    assert not X.empty
    assert not y.empty
    assert len(X) == len(y)

def test_train_model():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert model is not None
    assert hasattr(model, 'predict')

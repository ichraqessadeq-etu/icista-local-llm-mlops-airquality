import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test data loading."""
    data = load_data('data/air_quality.csv')
    assert not data.empty

def test_preprocess_data():
    """Test data preprocessing."""
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    assert not X.empty
    assert not y.empty

def test_train_model():
    """Test model training."""
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert model is not None

import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test data loading."""
    data = load_data('data/air_quality.csv')
    assert isinstance(data, pd.DataFrame)
    assert not data.empty

def test_preprocess_data():
    """Test data preprocessing."""
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert not X.empty
    assert not y.empty

def test_train_model():
    """Test model training."""
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100
    assert model.random_state == 42

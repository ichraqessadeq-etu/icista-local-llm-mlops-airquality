import pytest
from pipeline import load_data, preprocess_data, train_model, log_model

def test_load_data():
    """Test loading data from a CSV file."""
    data = load_data('data/air_quality.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing the data."""
    data = load_data('data/air_quality.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    """Test training the model."""
    data = load_data('data/air_quality.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_log_model():
    """Test logging the model using MLflow."""
    data = load_data('data/air_quality.csv')
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    log_model(model, 'AirQualityAnomalyDetection', 'Run1')

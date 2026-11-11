import pytest
from pipeline import load_data, preprocess_data, train_model, predict_anomalies

def test_load_data():
    """Test loading data from a CSV file."""
    file_path = 'data/air_quality.csv'
    data = load_data(file_path)
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing the data."""
    file_path = 'data/air_quality.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_train_model():
    """Test training the model."""
    file_path = 'data/air_quality.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

def test_predict_anomalies():
    """Test predicting anomalies."""
    file_path = 'data/air_quality.csv'
    data = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_test)
    predictions = predict_anomalies(model, X_test)
    assert isinstance(predictions, np.ndarray)

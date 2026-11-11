import pytest
import pipeline

def test_load_data():
    """Test loading the dataset."""
    data = pipeline.load_data('data/air_quality.csv')
    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame."

def test_preprocess_data():
    """Test preprocessing the data."""
    data = pipeline.load_data('data/air_quality.csv')
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(data)
    assert isinstance(X_train, pd.DataFrame), "X_train should be a pandas DataFrame."
    assert isinstance(X_test, pd.DataFrame), "X_test should be a pandas DataFrame."
    assert isinstance(y_train, pd.Series), "y_train should be a pandas Series."
    assert isinstance(y_test, pd.Series), "y_test should be a pandas Series."

def test_train_model():
    """Test training the model."""
    data = pipeline.load_data('data/air_quality.csv')
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(data)
    model = pipeline.train_model(X_train, y_train)
    assert isinstance(model, pipeline.RandomForestClassifier), "Model should be a RandomForestClassifier."

def test_evaluate_model():
    """Test evaluating the model."""
    data = pipeline.load_data('data/air_quality.csv')
    X_train, X_test, y_train, y_test = pipeline.preprocess_data(data)
    model = pipeline.train_model(X_train, y_train)
    accuracy = pipeline.evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float), "Accuracy should be a float."

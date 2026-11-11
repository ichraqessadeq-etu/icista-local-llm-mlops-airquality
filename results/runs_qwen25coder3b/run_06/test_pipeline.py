import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline

def test_load_data():
    """Test the load_data function."""
    data = load_data('data/air_quality.csv')
    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame."

def test_preprocess_data():
    """Test the preprocess_data function."""
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame."
    assert isinstance(y, pd.Series), "y should be a pandas Series."

def test_train_model():
    """Test the train_model function."""
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier."

def test_evaluate_model():
    """Test the evaluate_model function."""
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    assert isinstance(accuracy, float), "Accuracy should be a float."

def test_run_pipeline():
    """Test the run_pipeline function."""
    run_pipeline('data/air_quality.csv')

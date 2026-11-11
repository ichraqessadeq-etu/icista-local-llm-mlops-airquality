import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    """Test the load_data function."""
    try:
        data = load_data('data/air_quality.csv')
        assert isinstance(data, pd.DataFrame)
    except Exception as e:
        pytest.fail(f"Failed to load data: {e}")

def test_preprocess_data():
    """Test the preprocess_data function."""
    try:
        data = load_data('data/air_quality.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
    except Exception as e:
        pytest.fail(f"Failed to preprocess data: {e}")

def test_train_model():
    """Test the train_model function."""
    try:
        data = load_data('data/air_quality.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        assert isinstance(model, RandomForestClassifier)
    except Exception as e:
        pytest.fail(f"Failed to train model: {e}")

def test_evaluate_model():
    """Test the evaluate_model function."""
    try:
        data = load_data('data/air_quality.csv')
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
    except Exception as e:
        pytest.fail(f"Failed to evaluate model: {e}")

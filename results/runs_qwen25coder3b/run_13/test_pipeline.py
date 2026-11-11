import pytest
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    """Test the load_data function."""
    try:
        data = load_data('data/air_quality.csv')
        assert isinstance(data, pd.DataFrame)
        logging.info("load_data function test passed.")
    except Exception as e:
        logging.error(f"Test failed: {e}")
        pytest.fail()

def test_preprocess_data():
    """Test the preprocess_data function."""
    try:
        data = load_data('data/air_quality.csv')
        X_train, y_train, X_test, y_test = preprocess_data(data)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_test, pd.Series)
        logging.info("preprocess_data function test passed.")
    except Exception as e:
        logging.error(f"Test failed: {e}")
        pytest.fail()

def test_train_model():
    """Test the train_model function."""
    try:
        data = load_data('data/air_quality.csv')
        X_train, y_train, X_test, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        assert isinstance(model, RandomForestClassifier)
        logging.info("train_model function test passed.")
    except Exception as e:
        logging.error(f"Test failed: {e}")
        pytest.fail()

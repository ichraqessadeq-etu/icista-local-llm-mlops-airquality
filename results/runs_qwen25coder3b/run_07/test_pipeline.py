import pytest
import pipeline

def test_load_data():
    """Test loading data."""
    try:
        data = pipeline.load_data('data/air_quality.csv')
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
    except Exception as e:
        pytest.fail(f"Test failed: {e}")

def test_preprocess_data():
    """Test preprocessing data."""
    try:
        data = pipeline.load_data('data/air_quality.csv')
        X_train, X_test, y_train, y_test = pipeline.preprocess_data(data)
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
    except Exception as e:
        pytest.fail(f"Test failed: {e}")

def test_train_model():
    """Test training the model."""
    try:
        data = pipeline.load_data('data/air_quality.csv')
        X_train, X_test, y_train, y_test = pipeline.preprocess_data(data)
        model = pipeline.train_model(X_train, y_train)
        assert isinstance(model, pipeline.RandomForestClassifier)
    except Exception as e:
        pytest.fail(f"Test failed: {e}")

def test_evaluate_model():
    """Test evaluating the model."""
    try:
        data = pipeline.load_data('data/air_quality.csv')
        X_train, X_test, y_train, y_test = pipeline.preprocess_data(data)
        model = pipeline.train_model(X_train, y_train)
        pipeline.evaluate_model(model, X_test, y_test)
    except Exception as e:
        pytest.fail(f"Test failed: {e}")

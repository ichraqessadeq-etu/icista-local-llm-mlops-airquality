import pytest
from pipeline import load_data, preprocess_data, train_model
from mlflow.exceptions import MlflowException

def test_load_data():
    try:
        data = load_data('data/air_quality.csv')
        assert len(data) > 0
    except MlflowException as e:
        pytest.fail(f"Error loading data: {e}")

def test_preprocess_data():
    try:
        data = load_data('data/air_quality.csv')
        X, y = preprocess_data(data)
        assert len(X) > 0 and len(y) > 0
    except MlflowException as e:
        pytest.fail(f"Error preprocessing data: {e}")

def test_train_model():
    try:
        data = load_data('data/air_quality.csv')
        X, y = preprocess_data(data)
        model = train_model(X, y)
        assert model is not None
    except MlflowException as e:
        pytest.fail(f"Error training model: {e}")

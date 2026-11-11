import pytest
import pipeline

@pytest.fixture
def data():
    return pipeline.load_data('data/air_quality.csv')

def test_load_data(data):
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data(data):
    X, y = pipeline.preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    X, y = pipeline.preprocess_data(pipeline.load_data('data/air_quality.csv'))
    model = pipeline.train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    X, y = pipeline.preprocess_data(pipeline.load_data('data/air_quality.csv'))
    model = pipeline.train_model(X, y)
    accuracy = pipeline.evaluate_model(model, X, y)
    assert isinstance(accuracy, float)

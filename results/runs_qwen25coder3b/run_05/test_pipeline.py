import pytest
from pipeline import load_data, preprocess_data, train_model, run_pipeline

def test_load_data():
    """Test loading data."""
    with pytest.raises(MlflowException):
        load_data("non_existent_file.csv")

def test_preprocess_data():
    """Test preprocessing data."""
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'Anomaly': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape == (3, 10)
    assert y.shape == (3,)

def test_train_model():
    """Test training the model."""
    X = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'Anomaly': [0, 1, 0]
    })
    y = pd.Series([0, 1, 0])
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_run_pipeline():
    """Test running the pipeline."""
    with pytest.raises(MlflowException):
        run_pipeline("non_existent_file.csv")

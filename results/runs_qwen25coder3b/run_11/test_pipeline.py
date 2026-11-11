import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model, run_pipeline

def test_load_data():
    """Test loading data from a CSV file."""
    data = load_data('data/air_quality.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    """Test preprocessing the data."""
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'PT08.S2(NMHC)': [4, 5, 6],
        'Anomaly': [0, 1, 0]
    })
    features, target = preprocess_data(data)
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)

def test_train_model():
    """Test training the model."""
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'PT08.S2(NMHC)': [4, 5, 6],
        'Anomaly': [0, 1, 0]
    })
    features, target = preprocess_data(data)
    model = train_model(features, target)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    """Test evaluating the model."""
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'PT08.S2(NMHC)': [4, 5, 6],
        'Anomaly': [0, 1, 0]
    })
    features, target = preprocess_data(data)
    model = train_model(features, target)
    accuracy = evaluate_model(model, features, target)
    assert isinstance(accuracy, float)

def test_run_pipeline():
    """Test running the entire pipeline."""
    with pytest.raises(FileNotFoundError):
        run_pipeline('nonexistent_file.csv')

    with pytest.raises(KeyError):
        run_pipeline('data/air_quality.csv')

if __name__ == "__main__":
    pytest.main()

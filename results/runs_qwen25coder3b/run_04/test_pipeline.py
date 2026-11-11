import pytest
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    data = load_data('data/air_quality.csv')
    assert isinstance(data, pd.DataFrame)

def test_preprocess_data():
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'PT08.S2(NMHC)': [4, 5, 6],
        'Anomaly': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

def test_train_model():
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'PT08.S2(NMHC)': [4, 5, 6],
        'Anomaly': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_evaluate_model():
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'PT08.S2(NMHC)': [4, 5, 6],
        'Anomaly': [0, 1, 0]
    })
    X, y = preprocess_data(data)
    model = train_model(X, y)
    accuracy = evaluate_model(model, X, y)
    assert isinstance(accuracy, float)

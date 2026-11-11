import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    data = load_data('data/air_quality.csv')
    assert not data.empty
    assert 'Anomaly' in data.columns

def test_preprocess_data():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    assert X.shape[1] == 14  # 14 features
    assert y.shape[0] == X.shape[0]  # Same number of samples

def test_train_model():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    assert accuracy_score(y_test, y_pred) > 0.5  # Basic accuracy check

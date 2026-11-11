import pandas as pd
from pipeline import load_data, preprocess_data, train_model, evaluate_model

def test_load_data():
    data = load_data('data/air_quality.csv')
    assert not data.empty

def test_preprocess_data():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    assert X.shape[1] == 13  # 12 features + 1 target
    assert y.shape[0] == X.shape[0]

def test_train_model():
    data = load_data('data/air_quality.csv')
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    test_accuracy = evaluate_model(model, X_test, y_test)
    assert test_accuracy >= 0.5  # Assuming a reasonable threshold for accuracy

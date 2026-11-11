from pipeline import load_data, preprocess_data, train_model
import pytest
import pandas

def test_train_model():
    data = pandas.read_csv("data/air_quality.csv")
    X, y = preprocess_data(data)
    model = train_model(X, y)
    # Add assertions to check if the model is trained correctly

if __name__ == "__main__":
    test_train_model()

import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data("data/air_quality.csv")
        # Add assertions to check if the data is loaded correctly

    def test_preprocess_data(self):
        data = load_data("data/air_quality.csv")
        X, y = preprocess_data(data)
        # Add assertions to check if X and y are as expected

    def test_train_model(self):
        data = load_data("data/air_quality.csv")
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        # Add assertions to check if the model is trained correctly

if __name__ == "__main__":
    unittest.main()

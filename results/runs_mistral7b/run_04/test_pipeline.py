import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/air_quality.csv')
        # Add your assertions here

    def test_preprocess_data(self):
        data = load_data('data/air_quality.csv')
        X, y = preprocess_data(data)
        # Add your assertions here

    def test_train_model(self):
        data = load_data('data/air_quality.csv')
        X, y = preprocess_data(data)
        model = train_model(X, y)
        # Add your assertions here

if __name__ == '__main__':
    unittest.main()

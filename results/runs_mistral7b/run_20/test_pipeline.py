import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def test_load_data(self):
        data = load_data('data/air_quality.csv')
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data = load_data('data/air_quality.csv')
        X, y = preprocess_data(data)
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_train_model(self):
        data = load_data('data/air_quality.csv')
        X, y = preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        # Add assertions to check model performance here

if __name__ == '__main__':
    unittest.main()

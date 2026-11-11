import unittest
from pipeline import load_data, preprocess_data, train_model

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.data = load_data('data/air_quality.csv')
        self.X, self.y = preprocess_data(self.data)

    def test_train_model(self):
        model = train_model(self.X, self.y)
        # Add assertions to test the model's performance

if __name__ == '__main__':
    unittest.main()

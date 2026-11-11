import pytest
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    with pytest.raises(ValueError):
        load_data("non_existent_file.csv")

def test_preprocess_data():
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'PT08.S2(NMHC)': [4, 5, 6],
        'PT08.S3(NOx)': [7, 8, 9],
        'PT08.S4(NO2)': [10, 11, 12],
        'PT08.S5(O3)': [13, 14, 15],
        'NOx(GT)': [16, 17, 18],
        'C6H6(GT)': [19, 20, 21],
        'T': [22, 23, 24],
        'RH': [25, 26, 27],
        'AH': [28, 29, 30],
        'Hour': [31, 32, 33],
        'DayOfWeek': [34, 35, 36],
        'Month': [37, 38, 39],
        'Anomaly': [0, 1, 0]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    assert X_train.shape == (3, 16)
    assert y_train.shape == (3,)
    assert X_test.shape == (3, 16)
    assert y_test.shape == (3,)

def test_train_model():
    data = pd.DataFrame({
        'PT08.S1(CO)': [1, 2, 3],
        'PT08.S2(NMHC)': [4, 5, 6],
        'PT08.S3(NOx)': [7, 8, 9],
        'PT08.S4(NO2)': [10, 11, 12],
        'PT08.S5(O3)': [13, 14, 15],
        'NOx(GT)': [16, 17, 18],
        'C6H6(GT)': [19, 20, 21],
        'T': [22, 23, 24],
        'RH': [25, 26, 27],
        'AH': [28, 29, 30],
        'Hour': [31, 32, 33],
        'DayOfWeek': [34, 35, 36],
        'Month': [37, 38, 39],
        'Anomaly': [0, 1, 0]
    })
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)

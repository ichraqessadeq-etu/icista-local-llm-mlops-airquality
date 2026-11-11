import pandas as pd
from pipeline import load_data, preprocess_data, train_model

def test_load_data():
    data = load_data('data/air_quality.csv')
    assert not data.empty
    assert 'Anomaly' in data.columns

def test_preprocess_data():
    data = pd.DataFrame({
        'PT08.S1(CO)': [1.0, 2.0],
        'PT08.S2(NMHC)': [3.0, 4.0],
        'PT08.S3(NOx)': [5.0, 6.0],
        'PT08.S4(NO2)': [7.0, 8.0],
        'PT08.S5(O3)': [9.0, 10.0],
        'NOx(GT)': [11.0, 12.0],
        'C6H6(GT)': [13.0, 14.0],
        'T': [15.0, 16.0],
        'RH': [17.0, 18.0],
        'AH': [19.0, 20.0],
        'Hour': [21.0, 22.0],
        'DayOfWeek': [23.0, 24.0],
        'Month': [25.0, 26.0],
        'Anomaly': [0, 1]
    })
    X, y = preprocess_data(data)
    assert X.shape == (2, 14)
    assert y.shape == (2,)

def test_train_model():
    data = pd.DataFrame({
        'PT08.S1(CO)': [1.0, 2.0],
        'PT08.S2(NMHC)': [3.0, 4.0],
        'PT08.S3(NOx)': [5.0, 6.0],
        'PT08.S4(NO2)': [7.0, 8.0],
        'PT08.S5(O3)': [9.0, 10.0],
        'NOx(GT)': [11.0, 12.0],
        'C6H6(GT)': [13.0, 14.0],
        'T': [15.0, 16.0],
        'RH': [17.0, 18.0],
        'AH': [19.0, 20.0],
        'Hour': [21.0, 22.0],
        'DayOfWeek': [23.0, 24.0],
        'Month': [25.0, 26.0],
        'Anomaly': [0, 1]
    })
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)

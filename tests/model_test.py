import predict as predict
import pandas as pd

def test_prepare_df():
    patient = {
        'm/f': 1,
        'age': 99,
        'educ':2.0,
        'ses': 0.0,
        'etiv': 1500,
        'nwbv': 0.1,    
    }

    actual_df = predict.prepare_df(patient)

    expected_df = pd.DataFrame({
        'm/f': [1],
        'age': [99],
        'educ': [2.0],
        'ses': [0.0],
        'etiv': [1500],
        'nwbv': [0.1],
    })

    pd.testing.assert_frame_equal(actual_df, expected_df)

def test_get_and_apply_model():
    X_patient = pd.DataFrame({
        'm/f': [1],
        'age': [99],
        'educ': [2.0],
        'ses': [0.0],
        'etiv': [1500],
        'nwbv': [0.1],
    })
    
    actual_prediction = predict.get_and_apply_model(X_patient)

    expected_prediction = 1

    assert expected_prediction == actual_prediction
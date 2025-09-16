import pandas as pd
from app import get_model_parameters

def app_features(df):
    ''' Test app logic for handling complete / incomplete data'''
    # Get model parameters
    coefficients, _ = get_model_parameters()
    features = list(coefficients.keys())

    # Check for missing features
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        return "INCOMPLETE"
    else:
        return "COMPLETE"


def test_data_handling():
    """
    Test the data handling function with both a complete and incomplete DataFrame.
    """
    # Test with a complete DataFrame
    df_complete = pd.read_csv("sample_data_all.csv")
    assert app_features(df_complete) == "COMPLETE"

    # Test with an incomplete DataFrame
    df_incomplete = pd.read_csv("sample_data_missing.csv")
    assert app_features(df_incomplete) == "INCOMPLETE"
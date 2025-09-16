from app import get_model_parameters

def test_app_features(df):
    ''' Test app logic for handling complete / incomplete data'''
    # Get model parameters
    coefficients, intercept = get_model_parameters()
    features = list(coefficients.keys())

    # Ensure all required features are in the DataFrame
    if not all(feature in df.columns for feature in features):
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            return "INCOMPLETE"
        else:
            return "COMPLETE"


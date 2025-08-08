# src/predict.py

import joblib
import numpy as np
import pandas as pd
from . import config

def predict_new_sample(sample_df):
    """
    Takes a new sample as a pandas DataFrame and predicts its price using the saved model.
    :param sample_df: A pandas DataFrame with a single row containing the features.
    """
    # Load the model and scalers bundle
    bundle = joblib.load(config.MODEL_PATH)
    model = bundle['model']
    scaler_X = bundle['scaler_X']
    scaler_y = bundle['scaler_y']
    
    # In a real application, the feature names would be stored with the model.
    # We load them from the preprocessing step output for consistency.
    # For this script to be standalone, we can hard-code or load them.
    # Let's assume the DataFrame has the correct columns.
    
    sample_scaled = scaler_X.transform(sample_df)
    
    # Predict
    pred_scaled = model.predict(sample_scaled)
    
    # Inverse transform to get the original price scale
    pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
    
    return pred_original[0][0]

# Example of how to use this script
if __name__ == '__main__':
    # This is a hypothetical sample
    # The columns must match what the model was trained on
    sample_data = {
        'CRIM': [0.03], 'ZN': [0.0], 'INDUS': [7.07], 'CHAS': [0.0], 'NOX': [0.469],
        'RM': [6.421], 'AGE': [78.9], 'DIS': [4.9671], 'RAD': [2.0], 'TAX': [242.0],
        'PTRATIO': [17.8], 'B': [396.90], 'LSTAT': [9.14]
    }
    sample_df = pd.DataFrame(sample_data)
    
    predicted_price = predict_new_sample(sample_df)
    print(f'Predicted price for the sample: ${predicted_price * 1000:,.2f}')
# src/train.py

import xgboost as xgb
import joblib
import os
from . import config

def train_model(X_train, y_train):
    """Trains the XGBoost model using the training data."""
    model = xgb.XGBRegressor(**config.XGB_PARAMS)
    model.fit(X_train, y_train)
    print("XGBoost model trained successfully.")
    return model

def save_model_and_scalers(model, scaler_X, scaler_y):
    """Saves the trained model and scalers to a file."""
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    
    # Save model and scalers together in a dictionary
    joblib.dump({
        'model': model,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }, config.MODEL_PATH)
    
    print(f"Model and scalers saved successfully to: {config.MODEL_PATH}")
# src/evaluate.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_predictions(model, X_test):
    """Returns predictions for the test data."""
    return model.predict(X_test)

def calculate_metrics(y_true_scaled, y_pred_scaled, scaler_y, num_features):
    """Calculates and prints evaluation metrics."""
    # Inverse transform values to their original scale for interpretation
    y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    p = num_features
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    print("\n--- Model Evaluation ---")
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
    print(f'Mean Absolute Error (MAE): {mae:.4f}')
    print(f'R^2 Score: {r2:.4f}')
    print(f'Adjusted R^2 Score: {adjusted_r2:.4f}')
    
    # Return original-scale values for visualization
    return {
        'y_true_original': y_true,
        'y_pred_original': y_pred
    }
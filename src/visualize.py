# src/visualize.py

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'

def plot_actual_vs_predicted(y_true, y_pred):
    """Plots a scatter chart of actual vs. predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, color='red', alpha=0.5)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--') # Dashed black line
    plt.xlabel('Actual Values (in thousands of $)')
    plt.ylabel('Predicted Values (in thousands of $)')
    plt.title('Actual vs. Predicted House Values (Overall)')
    plt.grid(True)
    plt.show()

def plot_residuals(y_test_scaled, y_pred_scaled):
    """Plots the residuals to check for patterns."""
    residuals = y_test_scaled - y_pred_scaled
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred_scaled, residuals, color='purple', s=30, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Predicted Values (Scaled)')
    plt.ylabel('Residuals (Scaled)')
    plt.title('Residual Plot (XGBoost)')
    plt.grid(True)
    plt.show()

def plot_feature_importance(model, feature_names):
    """Plots the feature importance scores from the trained model."""
    importance = model.feature_importances_
    sorted_indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 7))
    plt.title("XGBoost Feature Importance")
    plt.bar(range(len(feature_names)), importance[sorted_indices], align="center")
    plt.xticks(range(len(feature_names)), np.array(feature_names)[sorted_indices], rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.show()
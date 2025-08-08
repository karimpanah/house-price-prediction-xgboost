# src/config.py

# File Paths
DATA_PATH = 'data/boston.csv'
MODEL_PATH = 'trained_models/xgboost_model.joblib'

# Target Column
TARGET_COLUMN = 'MEDV'

# Feature Selection Threshold
LOW_CORRELATION_THRESHOLD = 0.1

# Data Split Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# XGBoost Model Parameters
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'random_state': RANDOM_STATE
}

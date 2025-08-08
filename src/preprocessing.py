# src/preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from . import config

def load_and_clean_data(path):
    """Loads data from the path and fills missing values with the column mean."""
    df = pd.read_csv(path)
    
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_columns:
        if df[column].isnull().sum() > 0:
            df[column].fillna(df[column].mean(), inplace=True)
            
    return df

def select_features(df):
    """Selects features based on their correlation with the target variable."""
    correlation_matrix = df.corr()
    correlation_with_target = correlation_matrix[config.TARGET_COLUMN].sort_values(ascending=False)
    
    low_corr_features = correlation_with_target[abs(correlation_with_target) < config.LOW_CORRELATION_THRESHOLD].index.tolist()
    
    all_numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    feature_columns = [col for col in all_numeric_columns if col != config.TARGET_COLUMN and col not in low_corr_features]
    
    print(f"Features removed due to low correlation: {low_corr_features}")
    print(f"Final features selected: {feature_columns}")
    
    return feature_columns

def prepare_data():
    """Main function to run all preprocessing steps."""
    df = load_and_clean_data(config.DATA_PATH)
    feature_columns = select_features(df)
    
    X = df[feature_columns].values
    y = df[config.TARGET_COLUMN].values.reshape(-1, 1)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    # Scale the data (IMPORTANT: fit scalers only on training data)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train).flatten()
    y_test_scaled = scaler_y.transform(y_test).flatten()
    
    print("\nShape of processed data:")
    print(f"X_train: {X_train_scaled.shape}, y_train: {y_train_scaled.shape}")
    print(f"X_test: {X_test_scaled.shape}, y_test: {y_test_scaled.shape}")

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_scaled,
        'y_test': y_test_scaled,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'feature_columns': feature_columns
    }
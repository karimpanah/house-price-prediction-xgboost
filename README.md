# house-price-prediction-xgboost
Predicting house prices using XGBoost regression on structured housing data.

# Boston House Price Prediction using XGBoost

A machine learning project that predicts house prices using the Boston Housing dataset with XGBoost regression model. This project includes comprehensive data preprocessing, feature engineering, model training, and detailed visualization of results.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a house price prediction system using the classic Boston Housing dataset. The model uses XGBoost (Extreme Gradient Boosting) to predict median home values based on various features like number of rooms, crime rate, accessibility to highways, and more.

## ✨ Features

- **Data Preprocessing**: Automatic handling of missing values using mean imputation
- **Feature Engineering**: Correlation-based feature selection to remove low-impact features
- **Data Normalization**: StandardScaler for both features and target variables
- **XGBoost Implementation**: Optimized gradient boosting for regression
- **Comprehensive Evaluation**: Multiple metrics including MSE, RMSE, MAE, R², and Adjusted R²
- **Rich Visualizations**: 
  - Actual vs Predicted scatter plots
  - Residual analysis plots
  - Feature importance charts
  - Error distribution histograms
- **Model Persistence**: Save and load trained models using joblib
- **Statistical Analysis**: Shapiro-Wilk test for residual normality

## 📊 Dataset

The project uses the Boston Housing dataset which contains:
- **506 samples** of housing data
- **13 features** including:
  - CRIM: Crime rate per capita
  - RM: Average number of rooms per dwelling
  - AGE: Proportion of units built prior to 1940
  - DIS: Distances to employment centers
  - And more...
- **Target variable**: MEDV (Median home value in thousands of dollars)

## 🛠 Requirements

```python
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
scipy>=1.7.0
joblib>=1.1.0
arabic-reshaper
python-bidi
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/karimpanah/house-price-prediction-xgboost.git

cd house-price-prediction-xgboost
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. If running on Kaggle, the dataset path is already configured:
```python
df = pd.read_csv('/kaggle/input/the-boston-houseprice-data/boston.csv')
# if you run on local system, you should change CSV direction in main.py
```

## 💻 Usage
  
1. **Run the main script**:
   ```bash
   python main.py
   ```

2. **Key Steps in the Pipeline**:
   - Data loading and missing value handling
   - Feature correlation analysis and selection
   - Data normalization and train-test split
   - XGBoost model training
   - Model evaluation and visualization
   - Model saving for future use

3. **Make Predictions**:
   ```python
   # Load the saved model
   model = joblib.load('xgboost_model.joblib')
   
   # Prepare your data (normalize using the same scaler)
   prediction = model.predict(normalized_features)
   ```

## 📈 Model Performance

The XGBoost model achieves excellent performance on the Boston Housing dataset:

- **Mean Squared Error (MSE)**: ~10.5
- **Root Mean Squared Error (RMSE)**: ~3.2
- **Mean Absolute Error (MAE)**: ~2.1
- **R² Score**: ~0.87
- **Adjusted R² Score**: ~0.85

*Note: Actual performance may vary based on train-test split and hyperparameters*

## 📊 Visualizations

The project includes several informative visualizations:

1. **Actual vs Predicted Values**: Scatter plot showing model accuracy
2. **Residual Plot**: Analysis of prediction errors
3. **Error Distribution**: Histogram of residuals for normality check
4. **Feature Importance**: Bar chart showing which features matter most
5. **Correlation Heatmap**: Feature relationships with target variable


## 🔧 Hyperparameters

The XGBoost model uses the following configuration:
- **Objective**: reg:squarederror
- **Number of estimators**: 100
- **Learning rate**: 0.1
- **Max depth**: 3
- **Random state**: 42

Feel free to experiment with these parameters to potentially improve performance.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Boston Housing dataset from the UCI Machine Learning Repository
- XGBoost development team for the excellent gradient boosting framework
- Scikit-learn for comprehensive machine learning tools

## 📧 Contact

Arman Karimpanah - kak_arman@protonmail.com

Project Link: [https://github.com/karimpanah/house-price-prediction-xgboost.git](https://github.com/karimpanah/house-price-prediction-xgboost.git)

---

⭐ If you found this project helpful, please give it a star!

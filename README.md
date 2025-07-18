# Gold Price Prediction Using Random Forest Regressor

This project aims to predict the price of gold (GLD index) based on financial indicators using a machine learning model.

## Project Overview

- Dataset: gld_price_data.csv
- Model: Random Forest Regressor
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- Environment: Google Colab / Jupyter Notebook

## Dataset Description

- Rows: 2290 entries
- Features:
  - Date (ignored for modeling)
  - SPX (S&P 500 Index)
  - USO (United States Oil Fund)
  - SLV (Silver ETF)
  - EUR/USD (Euro to USD exchange rate)
- Target:
  - GLD (Gold ETF price)

## Workflow Summary

1. **Data Collection and Preprocessing**
   - Loaded dataset from CSV file
   - Verified no missing values
   - Analyzed distributions and summary statistics using `.describe()` and Seaborn’s `distplot` (deprecated)

2. **Feature Selection**
   - Dropped `Date` and target column `GLD`
   - Feature matrix `x`: SPX, USO, SLV, EUR/USD
   - Target vector `y`: GLD

3. **Train-Test Split**
   - Used `train_test_split()` to divide data:
     - 80% training
     - 20% testing

4. **Model Training**
   - Used `RandomForestRegressor` from scikit-learn with `n_estimators=100`
   - Model trained on training data using `fit()`

5. **Model Evaluation**
   - **Training set R² score**: 0.9984
   - **Testing set R² score**: 0.9888
   - Visualized actual vs predicted gold prices using `matplotlib`

6. **Conclusion**
   - The model demonstrates extremely high accuracy
   - Suitable for predicting gold prices based on associated financial market indicators


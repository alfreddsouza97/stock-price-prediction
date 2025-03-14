# Stock Price Prediction Using Machine Learning

## Overview

This Python script predicts future stock prices using historical data. It utilizes machine learning models (Linear Regression and Random Forest) to forecast stock prices for a given ticker symbol. The script fetches historical stock data from Yahoo Finance, preprocesses it, trains a predictive model, and estimates the stock price for a future date.

## Features

-   Fetches 15 years of historical stock data from Yahoo Finance.
    
-   Extracts relevant features (date, moving averages, etc.).
    
-   Supports two machine learning models:
    
    -   **Linear Regression**
        
    -   **Random Forest Regressor**
        
-   Evaluates the model using RMSE and R-squared metrics.
    
-   Predicts stock price for a given number of years into the future.
    
-   Optionally plots historical data with the predicted price.
    
-   Saves the trained model using joblib.
    

## Requirements

Ensure you have the following Python libraries installed before running the script:

```
pip install yfinance pandas scikit-learn numpy matplotlib joblib
```

## Usage

### Running the script

```
python 1.py
```

### Parameters

-   `ticker_symbol`: Stock ticker (e.g., `'AAPL'` for Apple, `'REDINGTON.NS'` for Redington India on NSE).
    
-   `prediction_years`: Number of years into the future to predict.
    
-   `model_choice`: Machine learning model (`'linear'` or `'random_forest'`).
    

### Example Code Usage

```
if __name__ == '__main__':
    ticker_symbol = 'DATAMATICS.NS'  # Stock ticker symbol
    prediction_years = 10   # Prediction time frame
    model_choice = 'random_forest'  # Model type: 'linear' or 'random_forest'

    predicted_price, rmse, r2, historical_data, trained_model = predict_stock_price(ticker_symbol, prediction_years, model_choice)
```

## Output

-   Prints model evaluation metrics (RMSE, R-squared).
    
-   Displays a plot of historical vs. predicted price (if matplotlib is installed).
    
-   Saves the trained model as `{ticker_symbol}_stock_model.pkl`.
    

## Dependencies

-   `yfinance`: Fetches historical stock data.
    
-   `pandas`: Data manipulation.
    
-   `scikit-learn`: Machine learning models.
    
-   `numpy`: Numerical operations.
    
-   `matplotlib`: Plotting (optional).
    
-   `joblib`: Model saving and loading.
    

## Model Training & Evaluation

-   **Training Data**: Extracted from Yahoo Finance.
    
-   **Features**: Date components (Year, Month, Day), moving averages (50-day, 200-day).
    
-   **Metrics**:
    
    -   **RMSE (Root Mean Squared Error)**: Measures prediction accuracy.
        
    -   **R-squared**: Indicates how well the model fits historical data.
        

## Saving & Loading the Model

### Saving the Model

```
import joblib
joblib.dump(trained_model, 'stock_model.pkl')
```

### Loading the Model Later

```
loaded_model = joblib.load('stock_model.pkl')
predicted_price = loaded_model.predict(new_data)
```

## Notes

-   Ensure you have an active internet connection for data fetching.
    
-   Moving averages are used as features but may not be optimal for long-term forecasting.
    
-   The model does not account for external factors like economic changes, news, or global events.
    

## Author

Alfred Marshall Dsouza

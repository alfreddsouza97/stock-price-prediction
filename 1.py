import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  # For a more complex model
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import datetime

def predict_stock_price(ticker, prediction_years=2, model_type='linear'):
    """
    Predicts the stock price for a given ticker in a specified number of years.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'REDINGTON.NS' for Redington India on NSE).
        prediction_years (int): The number of years into the future to predict.
        model_type (str):  The type of model to use: 'linear' (Linear Regression) or 'random_forest'.

    Returns:
        tuple: (predicted_price, rmse, r2, historical_data, model).
               Returns (None, None, None, None, None) if there's an error.
               predicted_price: The predicted stock price, or None if prediction failed.
               rmse: Root Mean Squared Error of the model on the test set.
               r2: R-squared value of the model on the test set.
               historical_data: The downloaded historical data (for plotting, etc.).
               model: The trained model.
    """
    try:
        # Download historical data
        data = yf.download(ticker, period="15y")  # Download 15 years of data

        if data.empty:
            print(f"Error: No data found for ticker symbol: {ticker}")
            return None, None, None, None, None

        # Feature Engineering:  Create features for the model
        data['Date'] = data.index
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['DayOfYear'] = data['Date'].dt.dayofyear
         # Add more features if needed (e.g., moving averages, technical indicators)
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        data.dropna(inplace=True) # Drop rows with NaN values (due to moving averages)


        # Prepare data for the model
        features = ['Year', 'Month', 'Day', 'DayOfYear','MA50','MA200']  # Use engineered features
        X = data[features]
        y = data['Close']  # Target variable is the closing price


        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% train, 20% test

        # Choose and train the model
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)  # Adjust parameters as needed
        else:
            print("Error: Invalid model_type. Choose 'linear' or 'random_forest'.")
            return None, None, None, None, None

        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Model Evaluation (on test set):\nRMSE: {rmse:.2f}\nR-squared: {r2:.2f}")



        # Make the prediction
        # 1. Get the last date in the historical data
        last_date = data['Date'].iloc[-1]
        # 2. Create a future date for the prediction
        future_date = last_date + pd.DateOffset(years=prediction_years)
        # 3. Create a DataFrame for the future date with the same features as the training data

        future_data = pd.DataFrame({
            'Year': [future_date.year],
            'Month': [future_date.month],
            'Day': [future_date.day],
            'DayOfYear': [future_date.dayofyear],
             # For moving averages, you ideally would project them, which is complex.
            # Here, we'll use the last available values as a simple (but less accurate) approach.
            'MA50': [data['MA50'].iloc[-1]],
            'MA200': [data['MA200'].iloc[-1]]
        })

        # 4. Predict the price using the trained model
        predicted_price = model.predict(future_data)[0]
        print(f"Predicted price in {prediction_years} years: {predicted_price:.2f}")

        return predicted_price, rmse, r2, data, model

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None, None, None


if __name__ == '__main__':
    ticker_symbol = 'DATAMATICS.NS'  # Redington India on NSE
    # ticker_symbol = 'AAPL' # Example for Apple
    prediction_years = 10   # change the prediction years accordingly
    model_choice = 'random_forest'  # or 'linear'

    predicted_price, rmse, r2, historical_data, trained_model = predict_stock_price(ticker_symbol, prediction_years, model_choice)


    if predicted_price is not None:
        # --- Plotting (Optional, requires matplotlib) ---
        try:
            import matplotlib.pyplot as plt

            # Create a new figure and axes
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot historical closing prices
            ax.plot(historical_data['Date'], historical_data['Close'], label='Historical Close Price')

            # Plot the predicted price as a single point
            future_date = historical_data['Date'].iloc[-1] + pd.DateOffset(years=prediction_years)
            ax.plot(future_date, predicted_price, 'ro', label=f'Predicted Price in {prediction_years} Years')

            # Customize the plot
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.set_title(f'{ticker_symbol} Stock Price Prediction')
            ax.legend()
            ax.grid(True)

            # Rotate date labels for better readability
            plt.xticks(rotation=45)
            plt.tight_layout()  # Adjust layout to prevent labels from overlapping
            plt.show()

        except ImportError:
            print("Matplotlib is not installed.  Plotting is skipped.")

    # Example of how to save the model (using joblib):
    try:
        import joblib
        joblib.dump(trained_model, f"{ticker_symbol}_stock_model.pkl")
        print(f"Model saved as {ticker_symbol}_stock_model.pkl")

        #  Load the model later:
        #  loaded_model = joblib.load(f"{ticker_symbol}_stock_model.pkl")
    except ImportError:
        print("Joblib is not installed. Model saving is skipped.")
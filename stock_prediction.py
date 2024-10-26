# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import os
import datetime as dt
import tensorflow as tf
import mplfinance as mpf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, LSTM, Dense, Dropout, Bidirectional, GRU, SimpleRNN
from constants import *
from Models import *

import yfinance as yf

# Get the data for the stock AAPL
data = yf.download(COMPANY,TRAIN_START,TRAIN_END)



def load_interest_data():
    # Load the interest rate data and parse dates
    interest_df = pd.read_excel(
        'data/interest_rates.xlsx',
        sheet_name='Data_Cleaned',
        usecols=['Date', 'Rate', 'Change']
    )
    # Rename columns for consistency
    interest_df.columns = ['Date', 'Change_In_Rate', 'Interest_Rate']
    interest_df['Date'] = pd.to_datetime(interest_df['Date'])
    interest_df.set_index('Date', inplace=True)
    return interest_df

def add_interest_col(stock_df, interest_df):
    # Use forward fill to propagate the closest previous interest rate and change in rate to each date in stock_df
    stock_df['Interest_Rate'] = interest_df['Interest_Rate'].reindex(stock_df.index, method='ffill')
    stock_df['Change_In_Rate'] = interest_df['Change_In_Rate'].reindex(stock_df.index, method='ffill')
    return stock_df


def clean_stock_data(ticker, start_date, end_date, handle_nan='drop', 
                     scale=False, feature_columns=None, save_local=True, 
                     load_local=True, local_dir='data', interest_rate_path='interest_rates.xlsx'):
    
    # Local file path
    local_file = os.path.join(local_dir, f"{ticker}_{start_date}_{end_date}.csv")
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Load data from local storage if available
    if load_local and os.path.exists(local_file):
        df = pd.read_csv(local_file, index_col='Date', parse_dates=True)
        print(f"Loaded data from {local_file}.")
    else:
        # Download data using yfinance
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            raise ValueError(f"No data was downloaded for {ticker}. Please check the ticker symbol and date range.")
        
        # Save data locally if required
        if save_local:
            df.to_csv(local_file)
            print(f"Data saved to {local_file}.")
    
    # Calculate the mid-point of Open & Close prices
    df['Mid_Price'] = (df['Open'] + df['Close']) / 2
    
    # Load interest rate data and add it to the stock data
    interest_df = load_interest_data()
    df = add_interest_col(df, interest_df)
    
    # Handle NaN values
    if handle_nan == 'drop':
        df.dropna(inplace=True)
    elif handle_nan == 'fill':
        df.ffill(inplace=True)
    
    # Scaling features
    if scale:
        if feature_columns is None:
            feature_columns = df.columns.tolist()
        for column in feature_columns:
            mmx_scaler = MinMaxScaler()
            df[column] = mmx_scaler.fit_transform(df[[column]])
            SCALERS[column] = mmx_scaler

    # Return the cleaned DataFrame
    return df





data = clean_stock_data(
    ticker=COMPANY,
    start_date=TRAIN_START,
    end_date=TRAIN_END,
    handle_nan='fill',
    scale=True,
    feature_columns=FEATURE_COLUMNS
)

print(data)
# plot a candlestick chart given a DataFrame
def plot_candlestick_chart(data, title="Candlestick Chart", n_days=1):

    # Resample the data over the specified number of days if n_days > 1        
    # # as there is no need to resample data to a 1 day timeframe
    # data displayed in each candlestick according to high, low, open and close categories
    # with width of each candle
    if n_days > 1:
        data_resampled = data.resample(f'{n_days}D').agg({
            'Open': 'first', 
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
    else:
        data_resampled = data
    
    # plot the chart using mplfinance
    mpf.plot(data_resampled, type='candle', volume=True, title=title, style='charles')


# plot_candlestick_chart(data,n_days=30)

# plot a boxplot chart given a DataFrame
def plot_boxplot_chart(data, title="Stock Prices Boxplot", n_days=1, columns=['Open', 'High', 'Low', 'Close']):

    # Resample the data over the specified number of days if n_days > 1
    # as there is no need to resample data to a 1 day timeframe
    # data categorized based on open, high, low and close categories
    if n_days > 1:
        data_resampled = data.resample(f'{n_days}D').agg({
            'Open': 'first', 
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        })
    else:
        data_resampled = data

    # Create a DataFrame for plotting, containing only the defined columns
    prices = data_resampled[columns]

    # Plot the boxplot using plt
    plt.figure(figsize=(10, 6))

    # create boxplot from resampled data
    prices.boxplot(column=columns)

    # define title of chart
    plt.title(title)

    # define labels for x and y axis
    plt.ylabel("Price")
    plt.xlabel("Price Categories")
   # plt.show()

# plot_boxplot_chart(data, n_days=30, columns=["High", "Low", "Open", "Close"])







# make combined/averaged predictions
def ensemble_predictions(models, k_steps):
    predictions = []

    # Loop through each model in the models list
    total_weight = 0
    for model_dict in models:
        predictions_model = model_dict["predictions"]      
        model_type = model_dict['type']
        weight = model_dict.get("weight", 1.0)
        total_weight += weight
        
        # Convert predictions to np array and apply weight
        predictions_model = np.array(predictions_model).reshape(-1)
        print(f"{model_type} predictions (first {k_steps} steps): {predictions_model[:k_steps]}")

        # Store the weighted predictions
        predictions.append(weight * predictions_model[:k_steps])



    # Combine predictions (weighted sum)
    combined_predictions = np.sum(predictions, axis=0) / total_weight

    return combined_predictions



prediction_columns = [
    {'Days': 30, 'Columns': FEATURE_COLUMNS},
]

layer_config = [
    {"type": "LSTM", "units": 128, "dropout": 0.3, "bidirectional": True},
    {"type": "GRU", "units": 64, "dropout": 0.2, "bidirectional": False},
    {"type": "Dense", "units": 32, "activation": "relu"}
]



# Initialize the Neural Network Model (LSTM/GRU/RNN)
nn_model = NeuralNetworkModel(data=data, sequence_length=SEQUENCE_LENGTH, n_features=len(FEATURE_COLUMNS), layer_config=layer_config)

# Train the model
nn_model.train(sequence_length=SEQUENCE_LENGTH, batch_size=32, epochs=50)




# Calculate the middle index of the data
middle_idx = len(data) // 2

# Get the start date from the index at middle_idx
start_date = data.index[middle_idx]

for config in prediction_columns:
    k_days = config['Days']
    feature_columns = config['Columns']

    # Update n_features in the model if FEATURE_COLUMNS change
    nn_model.n_features = len(FEATURE_COLUMNS)

    # Perform prediction using the predict_multistep method starting from middle_idx
    predictions = nn_model.predict_multistep(k_days, start_idx=middle_idx)

    # Check if predictions were generated
    if predictions.size == 0:
        print(f"No predictions were generated for {feature_columns}. Skipping plot.")
        continue

    # Print Predictions
    print(f"\nPredictions for the next {k_days} days starting from {start_date.date()} using columns {feature_columns}:")
    for i, prediction in enumerate(predictions):
        predicted_value = prediction.item()
        print(f"Day {i+1:<7} {predicted_value:<30.4f}")

    if False:
        # Inverse transform the actual data to bring it back to the original scale
        actual_scaled = data[PREDICTION_COLUMN].values.reshape(-1, 1)  # Reshape to 2D for inverse_transform
        actual = SCALERS[PREDICTION_COLUMN].inverse_transform(actual_scaled).flatten()  # Flatten back to 1D after scaling

        # Take the actual data from middle_idx to middle_idx + k_days for comparison
        actual_future_days = actual[middle_idx:middle_idx + k_days]

        # Create a date range for plotting
        prediction_dates = data.index[middle_idx:middle_idx + k_days]

        plt.figure(figsize=(10, 6))

        # Plot the actual data
        plt.plot(prediction_dates, actual_future_days, label="Actual")

        # Plot predictions
        plt.plot(prediction_dates, predictions.flatten(), label="Predicted", linestyle='--')

        # Customize the plot
        plt.title(f"Prediction vs Actual Data {feature_columns} features, {k_days} days starting from {start_date.date()})")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.legend()

        plt.show()




# Create sequences for multivariate & multistep prediction
X, y = [], []
for i in range(len(data) - SEQUENCE_LENGTH - K_STEPS + 1):
    X.append(data[FEATURE_COLUMNS].values[i:i + SEQUENCE_LENGTH])  # Multivariate input
    y.append(data['Close'].values[i + SEQUENCE_LENGTH:i + SEQUENCE_LENGTH + K_STEPS])  # Multistep output

X = np.array(X)
y = np.array(y)


# train SARIMA Model
sarima_model = SarimaModel(data)
sarima_model.train()

# train Random Forest Model
X_train, X_val, y_train, y_val = train_test_split(np.array(X), np.array(y), test_size=0.2, random_state=42, shuffle=False)
rf_model = RandomForestModel(X_train=X_train, y_train=y_train, X_test=X_val)
rf_model.train()



# make predictions
lstm_predictions = nn_model.predict(K_STEPS)
sarima_predictions = sarima_model.predict(K_STEPS)
rf_predictions = rf_model.predict(K_STEPS)

# Store the final 7 days of input data for the Close column
last_steps = SCALERS[PREDICTION_COLUMN].inverse_transform(data[PREDICTION_COLUMN].values[-K_STEPS:].reshape(-1, 1)).reshape(-1)


models = [
    {"type": "LSTM", "predictions": lstm_predictions, "weight": 0.8},
    {"type": "SARIMA", "predictions": sarima_predictions, "weight": 0.4},
    {"type": "Random Forest", "predictions": rf_predictions, "weight": 0.4}
]

# Get ensemble predictions
ensemble_pred = ensemble_predictions(models, K_STEPS)

#  print predictions
print(f"Last 7 days: {last_steps}")
print(f"Ensemble predictions: {ensemble_pred}")
# Ensure the actual values are sliced to the last K_STEPS
actual_last_steps = last_steps[-K_STEPS:]

# Create a time range for the x-axis
date_range = data[PREDICTION_COLUMN].index[-K_STEPS:] 



fig, ax1 = plt.subplots(figsize=(12, 6))
lines = []

line, = ax1.plot(date_range, actual_last_steps, label="Actual", marker='o')
lines.append(line)

# Plot the LSTM model's predicted values
line, = ax1.plot(date_range, lstm_predictions, label="LSTM Predictions", linestyle='--', marker='x')
lines.append(line)

# Plot the Ensemble model's predicted values
line, = ax1.plot(date_range, ensemble_pred, label="Ensemble Predictions", linestyle='-.', marker='s')
lines.append(line)

with_int = ""
if 'Interest_Rate' in FEATURE_COLUMNS:
    # Extract interest rates for the same date range
    interest_last_steps = data['Interest_Rate'][-K_STEPS:]
    # Convert to a NumPy array and reshape it before applying inverse transform
    interest_last_steps = SCALERS['Interest_Rate'].inverse_transform(interest_last_steps.values.reshape(-1, 1)).reshape(-1)

    # Plot interest rates on a secondary y-axis
    ax2 = ax1.twinx()  # Create a secondary y-axis
    line, = ax2.plot(date_range, interest_last_steps, label="Interest Rate", color='purple', linestyle=':', marker='d')
    lines.append(line)

    ax2.set_ylabel("Interest Rate (%)")
    with_int = " with Interest Rates"
elif 'Change_In_Rate' in FEATURE_COLUMNS:
    # Extract interest rates for the same date range
    interest_last_steps = data['Change_In_Rate'][-K_STEPS:]
    # Convert to a NumPy array and reshape it before applying inverse transform
    interest_last_steps = SCALERS['Change_In_Rate'].inverse_transform(interest_last_steps.values.reshape(-1, 1)).reshape(-1)

    # Plot interest rates on a secondary y-axis
    ax2 = ax1.twinx()  # Create a secondary y-axis
    line, = ax2.plot(date_range, interest_last_steps, label="Change_In_Rate Rate", color='purple', linestyle=':', marker='d')
    lines.append(line)

    ax2.set_ylabel("Change in Rate (%)")
    with_int = " with Change In Rates"


labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc="upper left")

# Customize the plot
plt.title(f"Actual vs. Predicted Values (LSTM and Ensemble Model){with_int}")
ax1.set_xlabel("Date")
ax1.set_ylabel("Closing Price")
ax1.grid(True)

# Show the plot
plt.show()



exit()
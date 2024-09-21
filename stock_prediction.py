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
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, LSTM, Dense, Dropout, Bidirectional, GRU, SimpleRNN


#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------
# DATA_SOURCE = "yahoo"
COMPANY = 'CBA.AX'

TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2023-08-01'       # End date to read

# data = web.DataReader(COMPANY, DATA_SOURCE, TRAIN_START, TRAIN_END) # Read data using yahoo


import yfinance as yf

# Get the data for the stock AAPL
data = yf.download(COMPANY,TRAIN_START,TRAIN_END)

#------------------------------------------------------------------------------
# Prepare Data
## To do:
# 1) Check if data has been prepared before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Use a different price value eg. mid-point of Open & Close
# 3) Change the Prediction days
#------------------------------------------------------------------------------

scaler = MinMaxScaler()


def clean_stock_data(ticker, start_date, end_date, handle_nan='drop', 
                     scale=False, feature_columns=None, save_local=True, 
                     load_local=True, local_dir='data'):
    
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
    
    print(df)
    # Calculate the mid-point of Open & Close prices
    df['Mid_Price'] = (df['Open'] + df['Close']) / 2
    
    # Handle NaN values
    if handle_nan == 'drop':
        df.dropna(inplace=True)
    elif handle_nan == 'fill':
        df.ffill(inplace=True)
    
    # Scaling features
    if scale:
        if feature_columns is None:
            feature_columns = df.columns.tolist()
        scalers = {}
        for column in feature_columns:
            if column in df.columns:
                df[column] = scaler.fit_transform(df[[column]])
                scalers[column] = scaler
            else:
                raise KeyError(f"Column '{column}' not found in data.")
    
    # Return the cleaned DataFrame
    return df


FEATURE_COLUMNS = ['Mid_Price','Close']
N_STEPS = 25


data = clean_stock_data(
    ticker=COMPANY,
    start_date='2020-01-01',
    end_date='2023-07-31',
    handle_nan='fill',
    scale=True,
    feature_columns=FEATURE_COLUMNS
)


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
    plt.show()

# plot_boxplot_chart(data, n_days=30, columns=["High", "Low", "Open", "Close"])


def create_model(sequence_length, n_features, layer_config, loss="mean_absolute_error", optimizer="rmsprop"):

    # Initialize the model
    model = Sequential()

    # Iterate through each passed layer config
    for i, layer in enumerate(layer_config):
        # Extract layer type and parameters passed
        layer_type = layer["type"]
        units = layer.get("units", 64)
        bidirectional = layer.get("bidirectional", False)
        dropout = layer.get("dropout", 0.0)
        activation = layer.get("activation", None)

        # Define the cell type based on input layer_type
        if layer_type == "LSTM":
            cell = LSTM
        elif layer_type == "GRU":
            cell = GRU
        elif layer_type == "RNN":
            cell = SimpleRNN
        else:
            cell = None

        # Add the first recurrent layer with a batch_input_shape
        if i == 0 and cell:
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), 
                                        batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, 
                               batch_input_shape=(None, sequence_length, n_features)))
        # Hidden layers
        elif cell:  
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))

        # Add dense layer
        if layer_type == "Dense":
            model.add(Dense(units, activation=activation))
        
        # Add dropout if specified
        if dropout > 0:
            model.add(Dropout(dropout))

    # Add a final Dense layer with 1 unit for regression output
    model.add(Dense(1, activation="linear"))

    # Compile the model
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)

    return model


if False:
    # Declare layers
    layers = [
        {"type": "LSTM", "units": 256, "bidirectional": True, "dropout": 0.3},
        {"type": "GRU", "units": 128, "bidirectional": False, "dropout": 0.2},
        {"type": "Dense", "units": 64, "activation": "relu"}
    ]

    # Create model
    model = create_model(
        sequence_length=25,
        n_features=2,
        layer_config=layers
    )





def train_model(model, data, sequence_length, batch_size=32, epochs=10, validation_split=0.2):
    X, y = [], []
    for i in range(len(data[FEATURE_COLUMNS].values) - sequence_length):
        X.append(data[FEATURE_COLUMNS].values[i:i + sequence_length])  # Sequence of `sequence_length` time steps
        y.append(data[FEATURE_COLUMNS].values[i + sequence_length, 0])  # Predict the next value for the first feature (Mid_Price)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(np.array(X), np.array(y), test_size=validation_split, random_state=42, shuffle=False)

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )

    return model, history


if False:
    training_configs = [
        {"batch_size": 64, "epochs": 100},
        {"batch_size": 32, "epochs": 100},
        {"batch_size": 32, "epochs": 150}
    ]

    layer_configs = {
        "LSTM": [
            {"type": "LSTM", "units": 128, "dropout": 0.2, "bidirectional": False},
            {"type": "LSTM", "units": 64, "dropout": 0.2, "bidirectional": False},
            {"type": "Dense", "units": 32, "activation": "relu"}
        ],
        "GRU": [
            {"type": "GRU", "units": 128, "dropout": 0.2, "bidirectional": False},
            {"type": "GRU", "units": 64, "dropout": 0.2, "bidirectional": False},
            {"type": "Dense", "units": 32, "activation": "relu"}
        ],
        "SimpleRNN": [
            {"type": "RNN", "units": 128, "dropout": 0.2, "bidirectional": False},
            {"type": "RNN", "units": 64, "dropout": 0.2, "bidirectional": False},
            {"type": "Dense", "units": 32, "activation": "relu"}
        ]
    }

    # Store final validation losses for comparison
    final_validation_losses = []

    # run all training configs with all layer types
    for layer_config_type, layer_config in layer_configs.items():
        for config in training_configs:

            print(f"\nTraining {layer_config_type} Model")
            # Create the model
            model = create_model(
                sequence_length=N_STEPS,
                n_features=len(FEATURE_COLUMNS),
                layer_config=layer_config
            )
            
            # Train the model using the provided layers/config
            print(f"Training with batch size {config['batch_size']} and epochs {config['epochs']}")
            trained_model, training_history = train_model(
                model=model,
                data=data,
                sequence_length=N_STEPS,
                batch_size=config['batch_size'],
                epochs=config['epochs']
            )
            
            # Get the final validation loss and training loss values
            final_val_loss = training_history.history['val_loss'][-1]
            final_train_loss = training_history.history['loss'][-1]
            
            # Append results to the list
            final_validation_losses.append({
                "model": layer_config_type,
                "batch_size": config['batch_size'],
                "epochs": config['epochs'],
                "training_loss": final_train_loss,
                "validation_loss": final_val_loss
            })

    # Print the final training and validation losses for each model and configuration
    print("\nFinal Training and Validation Losses:")
    for result in final_validation_losses:
        print(f"Model: {result['model']}, Batch Size: {result['batch_size']}, Epochs: {result['epochs']}, "
            f"Training Loss: {result['training_loss']:.4f}, Validation Loss: {result['validation_loss']:.4f}")



# Number of days to predict into the future (multistep)
k_steps = 5  # Predict the next 5 days

# Sequence length (number of past days used for prediction)
sequence_length = 25

# Prepare the data using existing functions
data = clean_stock_data(
    ticker=COMPANY,
    start_date='2020-01-01',
    end_date='2023-07-31',
    handle_nan='fill',
    scale=True,
    feature_columns=FEATURE_COLUMNS
)

# Use the existing function to create sequences for multivariate, multistep prediction
X, y = [], []
for i in range(len(data) - sequence_length - k_steps + 1):
    X.append(data[FEATURE_COLUMNS].values[i:i + sequence_length])  # Multivariate input
    y.append(data['Close'].values[i + sequence_length:i + sequence_length + k_steps])  # Multistep output

X = np.array(X)
y = np.array(y)

# Define the model using the existing function
layer_config = [
    {"type": "LSTM", "units": 128, "dropout": 0.2, "bidirectional": False},
    {"type": "LSTM", "units": 64, "dropout": 0.2, "bidirectional": False},
    {"type": "Dense", "units": 32, "activation": "relu"}
]

# Create the model for multivariate, multistep prediction
model = create_model(sequence_length=sequence_length, n_features=len(FEATURE_COLUMNS), layer_config=layer_config)

# Train the model using the existing function
trained_model, training_history = train_model(
    model=model,
    data=data,
    sequence_length=sequence_length,
    batch_size=32,
    epochs=50
)

# Print the final validation loss
final_val_loss = training_history.history['val_loss'][-1]
print(f"Final Validation Loss: {final_val_loss:.4f}")


# Make predictions k_days into the future given sequence length and feature columns
def make_predictions(model, data, k_days, sequence_length, feature_columns, prediction_column="Close"):

    # Prepare the most recent sequence of data from feature columns as input for prediction
    last_sequence = data[feature_columns].values[-sequence_length:]
    # Reshape the sequence to fit the model's batch_input_shape
    last_sequence = np.expand_dims(last_sequence, axis=0)
    
    predictions = []

    # Store the final 7 days of input data for the Adj Close column
    final_7_days = scaler.inverse_transform(data[prediction_column].values[-7:].reshape(-1, 1))
    
    # Find the index of the closing price column in the feature columns
    closing_price_index = feature_columns.index(prediction_column)
    
    for i in range(k_days):
        # Predict the next closing price
        next_pred = model.predict(last_sequence, verbose=0)
        # Get the predicted closing price
        next_pred_value = next_pred[0, 0]         
        predictions.append(next_pred_value)
        
        # Shift the sequence forward by removing the first timestep and add the predicted closing price
        new_sequence = np.copy(last_sequence[:, 1:, :]) 
        
        # Replace the closing price in the new sequence with the predicted value
        last_timestep = last_sequence[:, -1:, :]
        
        # Dynamically update the column corresponding to the closing price
        last_timestep[0, 0, closing_price_index] = next_pred_value.item()
        
        # Join the updated last timestep to the new sequence along time axis
        new_sequence = np.concatenate([new_sequence, last_timestep], axis=1)
        
        # Update last_sequence for the next iteration
        last_sequence = new_sequence  
    
    # Convert the list of predictions to a numpy array
    predictions = np.array(predictions).reshape(-1, 1)
    
    # Inverse scale the predictions to get the actual closing prices
    predictions = scaler.inverse_transform(predictions)
    
    return predictions, final_7_days


prediction_columns = [
    {'Days': 7, 'Columns': ['Open', 'Close']},
    {'Days': 30, 'Columns': ['Adj Close', 'Close']},
    {'Days': 14, 'Columns': ['High', 'Low','Close']},
    {'Days': 21, 'Columns': ['Volume', 'Close']}
]

for config in prediction_columns:
    k_days = config['Days']
    feature_columns = config['Columns']

    # Perform prediction using the specified columns and k_days
    predictions, final_days = make_predictions(
        model=trained_model,
        data=data,
        k_days=k_days,
        sequence_length=sequence_length,
        feature_columns=feature_columns
    )

    print(f"\nPredictions for the next {k_days} days using columns {feature_columns}:")

    print(f"{'Day':<10}{'Predicted Closing Price':<30}")

    # Loop through the predictions and print each day and prediction
    for idx, obj in enumerate(final_days):
        day = obj.item()
        print(f"Day {(0-(len(final_days)-idx)):<7} {day:<30.4f}")

    # Loop through the predictions and print each day and prediction
    for i in range(k_days):
        predicted_value = predictions[i].item()
        print(f"Day {i+1:<7} {predicted_value:<30.4f}")


exit()


PRICE_VALUE = "Close"

scaler = MinMaxScaler(feature_range=(0, 1)) 
# Note that, by default, feature_range=(0, 1). Thus, if you want a different 
# feature_range (min,max) then you'll need to specify it here
scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1)) 
# Flatten and normalise the data
# First, we reshape a 1D array(n) to 2D array(n,1)
# We have to do that because sklearn.preprocessing.fit_transform()
# requires a 2D array
# Here n == len(scaled_data)
# Then, we scale the whole array to the range (0,1)
# The parameter -1 allows (np.)reshape to figure out the array size n automatically 
# values.reshape(-1, 1) 
# https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
# When reshaping an array, the new shape must contain the same number of elements 
# as the old shape, meaning the products of the two shapes' dimensions must be equal. 
# When using a -1, the dimension corresponding to the -1 will be the product of 
# the dimensions of the original array divided by the product of the dimensions 
# given to reshape so as to maintain the same number of elements.

# Number of days to look back to base the prediction
PREDICTION_DAYS = 60 # Original

# To store the training data
x_train = []
y_train = []

scaled_data = scaled_data[:,0] # Turn the 2D array back to a 1D array
# Prepare the data
for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x-PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# We now reshape x_train into a 3D array(p, q, 1); Note that x_train 
# is an array of p inputs with each input being a 2D array 

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# This is our first hidden layer which also spcifies an input layer. 
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'

# test_data = web.DataReader(COMPANY, DATA_SOURCE, TEST_START, TEST_END)

# test_data = yf.download(COMPANY,TEST_START,TEST_END)

# The above bug is the reason for the following line of code
# test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

model_inputs = scaler.transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period 
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------


real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??
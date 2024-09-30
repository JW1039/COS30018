from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking, LSTM, Dense, Dropout, Bidirectional, GRU, SimpleRNN
import numpy as np
from sklearn.model_selection import train_test_split
import pmdarima as pm
from abc import ABC, abstractmethod
from constants import *

class Model():

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass
        


class NeuralNetworkModel(Model):
    def __init__(self, data, sequence_length, n_features, layer_config, loss="mean_absolute_error", optimizer="rmsprop"):
        # Initialize the model
        model = Sequential()
        self.Data = data
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

        self.Model = model

    def train(self, sequence_length=25, batch_size=32, epochs=10, validation_split=0.2):
        X, y = [], []
        for i in range(len(self.Data[FEATURE_COLUMNS].values) - sequence_length):
            X.append(self.Data[FEATURE_COLUMNS].values[i:i + sequence_length])  # Sequence of `sequence_length` time steps
            y.append(self.Data[FEATURE_COLUMNS].values[i + sequence_length, 0])  # Predict the next value for the first feature (Mid_Price)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(np.array(X), np.array(y), test_size=validation_split, random_state=42, shuffle=False)

        # Train the model
        history = self.Model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val)
        )

        return history

    def predict(self, k_days):
        
        last_sequence = self.Data[FEATURE_COLUMNS].values[-SEQUENCE_LENGTH:]
        # Reshape the sequence to fit the model's batch_input_shape
        last_sequence = np.expand_dims(last_sequence, axis=0)
        
        predictions = []
        # Find the index of the closing price column in the feature columns
        closing_price_index = FEATURE_COLUMNS.index(PREDICTION_COLUMN)
        
        for i in range(k_days):
            # Predict the next closing price
            next_pred = self.Model.predict(last_sequence, verbose=0)
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
        predictions = SCALERS[PREDICTION_COLUMN].inverse_transform(predictions)
        
        return predictions
    

class SarimaModel(Model):
    def __init__(self, data):
        self.Data = data
        # create Sarima model with seasonal window of 1 week
        self.Model = pm.auto_arima(data['Close'], seasonal=True, m=7)

    # no training needed as fitting happens in pm.auto_arima
    def train(self):
        pass;

    def predict(self, k_days):
        predictions = self.Model.predict(n_periods=k_days)
        predictions = SCALERS['Close'].inverse_transform(np.array(predictions).reshape(-1, 1)).reshape(-1)
        return predictions


class RandomForestModel(Model):

    def __init__(self, X_train, y_train, X_test):
        # create model with number of decision trees to be used
        self.Model = RandomForestRegressor(n_estimators=200)
        self.y_train = y_train
        # Reshape X_train and X_test to 2D arrays
        self.X_train_rf =X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
        self.X_test_rf = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])


    def train(self):
        # Train using X_train_rf and y_train
        self.Model.fit(self.X_train_rf, self.y_train)  

    def predict(self, k_days):
        predictions = []
        # Copy the test data to avoid modifying the original data
        current_X_test = self.X_test_rf.copy()  
        # perform multistep prediction
        for _ in range(k_days):
            # Predict the next day's value
            next_day_prediction = self.Model.predict(current_X_test)
            
            # Inverse transform the prediction back to the original scale
            next_day_prediction_scaled = SCALERS['Close'].inverse_transform(next_day_prediction.reshape(-1, 1)).reshape(-1)
            predictions.append(next_day_prediction_scaled[0])

            # Update current_X_test with the predicted value
            new_input = next_day_prediction.reshape(1, -1) 
            # Shift the window: remove the first time step and append the new prediction - shift along time axis
            current_X_test = np.roll(current_X_test, -1, axis=1)
            # assign predicted value
            current_X_test[:, -1] = new_input[:, 0]

        return np.array(predictions)


    



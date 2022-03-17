from tracemalloc import start
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, r2_score
import yfinance as yf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import json
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class Score():
    def load(self, stock_ticker, start_date):
        """Load data from yfinance. end_date not required because we take the most recent data
        Args:
            stock-ticker (string): string symbol to uniquely identify a particular stock or index
            start_date (string): Starting date of the required dataset (YYYY-MM-DD)
        
        Returns:
            dataframe of historical stock prices of the stock 
            corresponding to stock ticker from start_date to end_date
        """
        unseen_data = yf.download(stock_ticker, start_date)
        return unseen_data

    def create_model(self, input_size):
        """Initialize an LSTM model
        Args:
            input_size (integer): The length of the input data
        
        Returns:
            LSTM model compiled to optimize for mean squared error using Adam
        """
        model=Sequential()
        model.add(LSTM(100,return_sequences=True,input_shape=(input_size,1)))
        model.add(LSTM(100, return_sequences = True))
        model.add(LSTM(100))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')
        return model

    def load_model_weights(self, model_weights_path, input_size):
        """Load model weights saved from training
        Args:
            model_weights_path (string) : location where the model is saved
            input_size (integer): The length of the input data
        
        Returns:
            Loaded model with training weights
        """
        model = self.create_model(input_size)
        model.load_weights(model_weights_path)
        return model

    def test_data(self, unseen_data):
        """Get unseen features and target
        Args:
            unseen_data (dataframe): Input data (unseen)
        Returns:
            unseen_X (dataframe): Features of the unseen data
            unseen_y (array-like): Target of the unseen data
        """
        # Split the data into attribute
        unseen_X = unseen_data["Close"]
        unseen_y = unseen_data["Close"].shift(-1)[:-1]
        return unseen_X, unseen_y

    def load_scaler(self, scaler_path):
        """Load scaler saved from training
        Args:
            scaler_path (string): Location of the saved scaler
        
        Returns:
            Saved instance of MinMaxScaler
        """
        scaler=joblib.load(scaler_path)
        return scaler

    def scale(self, unseen_X, scaler_path):
        """Scale the attributes
        Args:
            unseen_X (dataframe): Features of the unseen data
        
        Returns:
            Scaled version of:
                unseen_X
        """
        unseen_X = self.load_scaler(scaler_path).transform(np.array(unseen_X).reshape(-1,1))
        return unseen_X

    def predict_and_invert(self, model, unseen_X, scaler_path):
        """Initialize an LSTM model
        Args:
            unseen_X (dataframe): Features of the unseen_data
            scaler_path (string): Location of saved scaler
        Returns:
            Predictions on unseen_X after descaling
        """
        prediction = model.predict(unseen_X)
        scaler = self.load_scaler(scaler_path)
        pred = prediction
        pred= scaler.inverse_transform(pred)
        return pred

    def combine_results(self, unseen_data, unseen_y, pred):
        """Combine predictions with the input data to use for evaluation
        Args:
            unseen_data (dataframe): Data loaded from yfinance
            unseen_y (array-like): Target variable
            pred (array-like): Predictions on unseen_data

        Returns:
            Unseen data along with the predictions
        """
        unseen_data["OpenTom"] = unseen_data["Open"].shift(-1)
        unseen_data["y"] = unseen_y
        unseen_data["Pred"] = pred
        return unseen_data


    def __init__(self, stock_ticker, start_date, model_weights_path, scaler_path, output_data_path):
        """Loads unseen data, uses saved model to make predictions and saves the data
           for evaluation
        Args:
            stock-ticker (string): string symbol to uniquely identify a particular stock or index
            start_date (string): Starting date of the required dataset (YYYY-MM-DD)
            model_weights_path (string): Location of saved LSTM model
            scaler_path (string): Location of the saved fitted scaler
            output_data_path (string): Location to store combined data for evaluation
        """
        unseen_data = self.load(stock_ticker, start_date)
        model = self.load_model_weights(model_weights_path, unseen_data.shape[0])
        unseen_X, unseen_y = self.test_data(unseen_data)
        unseen_X = self.scale(unseen_X,scaler_path)
        pred = self.predict_and_invert(model, unseen_X, scaler_path)
        unseen_data = self.combine_results(unseen_data, unseen_y, pred)
        logging.info("Predictions made successfully")
        joblib.dump(unseen_data, output_data_path)

if __name__ == "__main__":
    f = open('config/params.json')
    params = json.load(f)

    stock_ticker = params["score"]["stock_ticker"]
    start_date = params["score"]["start_date"]
    model_weights_path = params["score"]["model_weights_path"]
    scaler_path = params["score"]["scaler_path"]
    output_data_path = params["score"]["output_data_path"]
    Score(stock_ticker, start_date, model_weights_path, scaler_path, output_data_path)



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

class Score():
    def load(self, stock_ticker, start_date):
        unseen_data = yf.download(stock_ticker, start_date)
        return unseen_data

    def create_model(self, input_size):
        model=Sequential()
        model.add(LSTM(100,return_sequences=True,input_shape=(input_size,1)))
        model.add(LSTM(100, return_sequences = True))
        model.add(LSTM(100))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')
        return model

    def load_model_weights(self, model_weights_path, input_size):
        model = self.create_model(input_size)
        model.load_weights(model_weights_path)
        return model

    def test_data(self, unseen_data):
        unseen_X = unseen_data["Close"]
        unseen_y = unseen_data["Close"].shift(-1)[:-1]
        return unseen_X, unseen_y

    def load_scaler(self, scaler_path):
        scaler=joblib.load(scaler_path)
        return scaler

    def scale(self, unseen_X, scaler_path):
        unseen_X = self.load_scaler(scaler_path).transform(np.array(unseen_X).reshape(-1,1))
        return unseen_X

    def predict_and_invert(self, model, unseen_X, scaler_path):
        prediction = model.predict(unseen_X)
        scaler = self.load_scaler(scaler_path)
        pred = prediction[:-1]
        pred= scaler.inverse_transform(pred)
        return pred

    def combine_results(self, unseen_data, unseen_y, pred):
        unseen_data["OpenTom"] = unseen_data["Open"].shift(-1)
        unseen_data = unseen_data.iloc[:-1,:]
        unseen_data["y"] = unseen_y
        unseen_data["Pred"] = pred
        return unseen_data
    
    def dump(self, unseen_data, output_data_path):
        joblib.dump(unseen_data, output_data_path)

    def __init__(self, stock_ticker, start_date, model_weights_path, scaler_path, output_data_path):
        unseen_data = self.load(stock_ticker, start_date)
        model = self.load_model_weights(model_weights_path, unseen_data.shape[0])
        unseen_X, unseen_y = self.test_data(unseen_data)
        scaler = self.load_scaler(scaler_path)
        unseen_X = self.scale(unseen_X,scaler_path)
        pred = self.predict_and_invert(model, unseen_X, scaler_path)
        unseen_data = self.combine_results(unseen_data, unseen_y, pred)
        logging.info("Predictions made successfully")
        self.dump(unseen_data, output_data_path)


if __name__ == "__main__":
    f = open('config/params.json')
    params = json.load(f)

    stock_ticker = params["score"]["stock_ticker"]
    start_date = params["score"]["start_date"]
    model_weights_path = params["score"]["model_weights_path"]
    scaler_path = params["score"]["scaler_path"]
    output_data_path = params["score"]["output_data_path"]
    Score(stock_ticker, start_date, model_weights_path, scaler_path, output_data_path)



import yfinance as yf
import joblib
import pandas as pd
import json

class Prepare():
    def load(self, stock_ticker, start_date, end_date):
        data = yf.download(stock_ticker, start_date, end_date)
        return data
    
    def preprocess(self,data):
        data = data[["Close"]]
        data["Target"] = data["Close"].shift(-1)
        return data

    def data_dump(self, data, data_path):
        joblib.dump(data, data_path)

    def __init__(self, stock_ticker, start_date, end_date, data_path):
        data = yf.download(stock_ticker, start_date, end_date)
        data = self.preprocess(data)
        self.data_dump(data, data_path)

if __name__ == "__main__":
    f = open('config/params.json')
    params = json.load(f)

    stock_ticker = params["prepare_data"]["stock_ticker"]
    start_date = params["prepare_data"]["start_date"]
    end_date = params["prepare_data"]["end_date"]
    data_path = params["prepare_data"]["data_path"]

    Prepare(stock_ticker, start_date, end_date, data_path)
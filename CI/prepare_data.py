import yfinance as yf
import joblib
import pandas as pd

stock_ticker = "ITC.NS"
start_date = "2020-01-01"
end_date = "2021-12-31"
data_path = "joblib/data.joblib"

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
    stock_ticker = "ITC.NS"
    start_date = "2020-01-01"
    end_date = "2021-12-31"
    data_path = "joblib/data.joblib"
    Prepare(stock_ticker, start_date, end_date, data_path)
import yfinance as yf
import joblib
import json

class Prepare():
    def load(self, stock_ticker, start_date, end_date):
        """Load data from yfinance
        Args:
            stock-ticker (string): string symbol to uniquely identify a particular stock or index
            start_date (string): Starting date of the required dataset (YYYY-MM-DD)
            end_date (string): Ending date of the required dataset (YYYY-MM-DD)
        
        Returns:
            dataframe of historical stock prices of the stock 
            corresponding to stock ticker from start_date to end_date
        """
        data = yf.download(stock_ticker, start_date, end_date)
        return data
    
    def preprocess(self,data):
        """Get the features and target variables
        Args:
            data (dataframe) : A dataframe of historical stock price data
        
        Returns:
            A dataframe containing the features and series of targets.
        """
        data = data[["Close"]]
        data["Target"] = data["Close"].shift(-1)
        return data

    def data_dump(self, data, data_path):
        """Store the data for further use
        Args:
            data (dataframe): A dataframe of historical stock price data
            data_path (string): Location where you want to save this dataframe
        """
        joblib.dump(data, data_path)

    def __init__(self, stock_ticker, start_date, end_date, data_path):
        """init function of the class Prepare
        Args:
            stock-ticker (string): string symbol to uniquely identify a particular stock or index
            start_date (string): Starting date of the required dataset (YYYY-MM-DD)
            end_date (string): Ending date of the required dataset (YYYY-MM-DD)
            data_path (string): Location where you want to save this dataframe
        """
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
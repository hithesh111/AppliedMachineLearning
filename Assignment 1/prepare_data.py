from calendar import weekday
from sys import prefix
import yfinance as yf
import joblib
import pandas as pd
from datetime import datetime

data = yf.download("ITC.NS",start = "2020-01-01", end = "2021-12-31")
# data["Target"] = (data["Close"] - data["Open"]).apply(lambda x: 1 if x>0 else 0).shift(-1)
# data = data[["Close","Open","Target"]]
data = data[["Close"]]
data["Target"] = data["Close"].shift(-1)
joblib.dump(data, "joblib/data.joblib")
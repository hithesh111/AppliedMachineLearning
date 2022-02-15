from cmath import nan
import yfinance as yf
import joblib

caplin_point = yf.download("CAPLIPOINT.NS", start = "2015-01-01", end = "2022-02-03")

caplin_point["Close-Open"] = caplin_point["Close"] - caplin_point["Open"]
caplin_point["Target"] = caplin_point["Close-Open"].apply(lambda x: 1 if x>0 else 0).shift(-1)

joblib.dump(caplin_point, "CAPLIN_POINT.joblib")
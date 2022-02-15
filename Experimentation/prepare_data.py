from cmath import nan
import yfinance as yf
import joblib

caplin_point = yf.download("CAPLIPOINT.NS", start = "2020-01-01", end = "2022-02-03")

y = list((caplin_point["Close"] - caplin_point["Open"]).apply(lambda x: 1 if x>0 else 0))
y_shifted = [float("nan")] + y[:-1]
caplin_point["Target"] = y_shifted

joblib.dump(caplin_point, "CAPLIN_POINT.joblib")
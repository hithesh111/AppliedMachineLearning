import yfinance as yf
import joblib

caplin_point = yf.download("CAPLIPOINT.NS", start = "2020-01-01", end = "2022-02-03")

joblib.dump(caplin_point, "CAPLIN_POINT.joblib")
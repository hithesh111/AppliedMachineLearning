from cmath import nan
import yfinance as yf
import joblib

dabur = yf.download("DABUR.NS", start = "2012-01-01", end = "2022-02-03")

dabur["Close-Open"] = dabur["Close"] - dabur["Open"]
dabur["Percent_Change"] = 100*dabur["Close-Open"]/dabur["Open"]
dabur["Volatility"] = 100*(dabur["High"] - dabur["Low"])/dabur["Open"] 
dabur["Target"] = dabur["Close-Open"].apply(lambda x: 1 if x>0 else 0).shift(-1)

n_days = 30

for i in range(1,n_days+1):
    dabur["PrevTarget_"+str(i)+"lag"] = dabur["Target"].shift(i)
    dabur["Close-Open_"+str(i)+"lag"] = dabur["Close-Open"].shift(i)
    dabur["Volatility_"+str(i)+"lag"] = dabur["Volatility"].shift(i)
    dabur["Percent_Change_"+str(i)+"lag"] = dabur["Percent_Change"].shift(i)

joblib.dump(dabur, "Experimentation/DABUR.joblib")
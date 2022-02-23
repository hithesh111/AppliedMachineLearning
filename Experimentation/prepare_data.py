import yfinance as yf
import joblib

data = yf.download("KSCL.BO")

data["Close-Open"] = data["Close"] - data["Open"]
data["Percent_Change"] = 100*data["Close-Open"]/data["Open"]
data["Volatility"] = 100*(data["High"] - data["Low"])/data["Open"] 
data["Target"] = data["Close-Open"].apply(lambda x: 1 if x>0 else 0).shift(-1)
data["SupRes1"] = data["Close"].apply(lambda x: 1 if x>610 else 0)
data["SupRes2"] = data["Close"].apply(lambda x: 1 if x>500 else 0)
data["SupRes3"] = data["Close"].apply(lambda x: 1 if x>300 else 0)

data['SMA50'] = data['Close'].rolling(50).mean()
data['SMA100'] = data['Close'].rolling(100).mean()
data['SMA200'] = data['Close'].rolling(200).mean()

data['SMM50'] = data['Close'].rolling(50).max()
data['SMM100'] = data['Close'].rolling(100).max()
data['SMM200'] = data['Close'].rolling(200).max()

data['SMm50'] = data['Close'].rolling(50).min()
data['SMm100'] = data['Close'].rolling(100).min()
data['SMm200'] = data['Close'].rolling(200).min()


n_days = 4

for i in range(1,n_days+1):
    data["PrevTarget_"+str(i)+"lag"] = data["Target"].shift(i)
    data["Close-Open_"+str(i)+"lag"] = data["Close-Open"].shift(i)
    data["Volatility_"+str(i)+"lag"] = data["Volatility"].shift(i)
    data["Percent_Change_"+str(i)+"lag"] = data["Percent_Change"].shift(i)
    data["SupRes1_"+str(i)+"lag"] = data["SupRes1"].shift(i)
    data["SupRes2_"+str(i)+"lag"] = data["SupRes2"].shift(i)
    data["SupRes3_"+str(i)+"lag"] = data["SupRes3"].shift(i)

print(data.sample(5))
joblib.dump(data, "Experimentation/data.joblib")
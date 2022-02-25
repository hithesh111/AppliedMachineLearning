from sys import prefix
import yfinance as yf
import joblib
import pandas as pd

data = yf.download("ITC.NS")

OHLCAV = ["Open","High","Low","Close","Adj Close", "Volume"]

data["Close-Open"] = data["Close"] - data["Open"]
data["Percent_Change"] = 100*data["Close-Open"]/data["Open"]
data["Volatility"] = 100*(data["High"] - data["Low"])/data["Open"]
data["Today"] = data["Close-Open"].apply(lambda x: 1 if x>0 else 0)
data["Target"] = data["Close-Open"].apply(lambda x: 1 if x>0 else 0).shift(-1)

data["Date"] = data.index

data["DayOTW"] = data["Date"].dt.dayofweek
onehot_dayOTW = pd.get_dummies(data["DayOTW"], prefix = "DayOTW")
data = pd.concat([data,onehot_dayOTW],axis = 1)

data["Day"] = data["Date"].dt.day
onehot_day = pd.get_dummies(data["Day"], prefix = "Day")
data = pd.concat([data,onehot_day],axis = 1)

data["Week"] = data["Day"].apply(lambda x: 1 + x//7)
onehot_week = pd.get_dummies(data["Week"], prefix = "Week")
data = pd.concat([data, onehot_week], axis = 1)

data["Month"] = data["Date"].dt.month
onehot_Month = pd.get_dummies(data["Month"], prefix = "Month")
data = pd.concat([data,onehot_Month],axis = 1)

data["Quarter"] = data["Month"].apply(lambda x: 1 + x//4)
onehot_quarter = pd.get_dummies(data["Quarter"], prefix = "Quarter")
data = pd.concat([data, onehot_quarter], axis = 1)

DATEDAYMONTH = ["Date", "DayOTW", "Day","Month", "Week", "Quarter"]

# data['SMA50'] = data['Close'].rolling(50).mean()
# data['SMA100'] = data['Close'].rolling(100).mean()
# data['SMA200'] = data['Close'].rolling(200).mean()

# data['SMM50'] = data['Close'].rolling(50).max()
# data['SMM100'] = data['Close'].rolling(100).max()
# data['SMM200'] = data['Close'].rolling(200).max()

# data['SMm50'] = data['Close'].rolling(50).min()
# data['SMm100'] = data['Close'].rolling(100).min()
# data['SMm200'] = data['Close'].rolling(200).min()

# data["SMA50Diff"] = 100*(data["Close"] - data["SMA50"])/data["SMA50"]
# data["SMA100Diff"] = 100*(data["Close"] - data["SMA100"])/data["SMA100"]
# data["SMA200Diff"] = 100*(data["Close"] - data["SMA200"])/data["SMA200"]

# data["SMM50Diff"] = 100*(data["Close"] - data["SMM50"])/data["SMM50"]
# data["SMM100Diff"] = 100*(data["Close"] - data["SMM100"])/data["SMM100"]
# data["SMM200Diff"] = 100*(data["Close"] - data["SMM200"])/data["SMM200"]

# data["SMm50Diff"] = 100*(data["Close"] - data["SMm50"])/data["SMm50"]
# data["SMm100Diff"] = 100*(data["Close"] - data["SMm100"])/data["SMm100"]
# data["SMm200Diff"] = 100*(data["Close"] - data["SMm200"])/data["SMm200"]

MovingAvgMaxMin = ["SMA50","SMA100","SMA200","SMM50","SMM100","SMM200","SMm50","SMm100","SMm200"]

n_days = 10
for i in range(2,n_days+1):
    data["Past"+str(i)+"DaysA"] = data["Today"].rolling(i).mean()
    data["Past"+str(i)+"DaysM"] = data["Today"].rolling(i).max()
    data["Past"+str(i)+"Daysm"] = data["Today"].rolling(i).min()

print(data["Past3DaysM"])

# data = data.drop(MovingAvgMaxMin,1)
data = data.drop(OHLCAV, axis = 1)
data = data.drop(DATEDAYMONTH, axis = 1)

n_days = 1
print(data.columns)
for i in range(1,n_days+1):
    data["PrevTarget_"+str(i)+"lag"] = data["Target"].shift(i).copy()
    data["Volatility_"+str(i)+"lag"] = data["Volatility"].shift(i).copy()
    data["Percent_Change_"+str(i)+"lag"] = data["Percent_Change"].shift(i).copy()

joblib.dump(data, "Experimentation/joblib/data.joblib")
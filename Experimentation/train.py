from re import X
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

data = joblib.load("joblib/data.joblib")
X = data.drop(["Target"],1).iloc[:-1,:]["Close"]
y = data["Target"][:-1]

split = int(0.8*data.shape[0])
X_train, X_test = X[:split], X.iloc[split:]
y_train, y_test = y[:split], y[split:]

scaler=MinMaxScaler(feature_range=(0,1))
X_train=scaler.fit_transform(np.array(X_train).reshape(-1,1))
X_test=scaler.fit_transform(np.array(X_test).reshape(-1,1))
y_train=scaler.fit_transform(np.array(y_train).reshape(-1,1))
y_test=scaler.fit_transform(np.array(y_test).reshape(-1,1))

def create_model():
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[0],1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model


model = create_model()

model.fit(X_train, y_train, epochs=80,validation_data=(X_test,y_test), verbose=1)

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

X_train=scaler.inverse_transform(X_train)
X_test=scaler.inverse_transform(X_test)
y_train=scaler.inverse_transform(y_train)
y_test=scaler.inverse_transform(y_test)

print(r2_score(train_predict,y_train))
print(r2_score(test_predict,y_test))

y_rec = np.concatenate([y_train,y_test],axis = 0)
y_pred = np.concatenate([train_predict, test_predict], axis= 0)
X_rec = np.concatenate([X_train,X_test],axis = 0)

final_data = pd.DataFrame(index = data.iloc[:-1,:].index)
final_data["Close"] = X_rec
final_data["Pred"] = y_pred
final_data["Target"] = y_rec

plt.plot(test_predict)
plt.savefig("pred.jpg")

plt.plot(y_test)
plt.savefig("test.jpg")

model.save_weights('save/')
joblib.dump(scaler,"joblib/scaler.joblib")
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, r2_score
import yfinance as yf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

unseen_data = yf.download("ITC.NS", start = "2022-01-01")

def create_model():
    model=Sequential()
    model.add(LSTM(100,return_sequences=True,input_shape=(unseen_data.shape[0],1)))
    model.add(LSTM(100, return_sequences = True))
    model.add(LSTM(100))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

model = create_model()
model.load_weights('save/')

unseen_X = unseen_data["Close"]
unseen_y = unseen_data["Close"].shift(-1)[:-1]

scaler=joblib.load("joblib/scaler.joblib")
unseen_X = scaler.transform(np.array(unseen_X).reshape(-1,1))
# unseen_y = scaler.transform(np.array(unseen_data["Close"].shift(-1)[:-1]).reshape(-1,1))

prediction = model.predict(unseen_X)
pred_for_tomorrow = scaler.inverse_transform(prediction[-1:])

pred = prediction[:-1]
pred= scaler.inverse_transform(pred)
# unseen_y = scaler.inverse_transform(unseen_y)

unseen_data["OpenTom"] = unseen_data["Open"].shift(-1)

unseen_data = unseen_data.iloc[:-1,:]
unseen_data["y"] = unseen_y
unseen_data["Pred"] = pred
joblib.dump(unseen_data, "joblib/unseen.joblib")


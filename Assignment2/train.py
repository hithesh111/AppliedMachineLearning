from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd
import json
import logging

class Training():
    def DataSplit(self,X,y):
        #Do a split of training and testing data
        split = int(0.8*X.shape[0])
        X_train, X_test = X[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]
        return X_train, X_test, y_train, y_test

    def Fit_Scaler(self,X_train,X_test,y_train,y_test, scaler):
        #Fit a scaler to training attributes and then scale the rest using same scaler
        X_train=scaler.fit_transform(np.array(X_train)).reshape(-1,1)
        X_test=scaler.transform(np.array(X_test)).reshape(-1,1)
        y_train=scaler.transform(np.array(y_train).reshape(-1,1))
        y_test=scaler.transform(np.array(y_test).reshape(-1,1))
        return X_train, X_test, y_train, y_test

    def create_model(self,input_length):
        #Create an empty LSTM model
        model=Sequential()
        model.add(LSTM(100,return_sequences=True,input_shape=(input_length,1)))
        model.add(LSTM(100, return_sequences = True))
        model.add(LSTM(100))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')
        return model

    def train_validate_predict(self,X_train,X_test,y_train,y_test, model):
        #Fit the model using training data and make predictions for cross validation
        model.fit(X_train, y_train, epochs=50,validation_data=(X_test,y_test), verbose=1)
        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)
        return model, train_predict, test_predict

    def ScalerInvert(self,scaler, X_train,X_test, y_train, y_test, train_predict, test_predict):
        #Invert the training/testing data as well as the predictions
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)
        X_train=scaler.inverse_transform(X_train)
        X_test=scaler.inverse_transform(X_test)
        y_train=scaler.inverse_transform(y_train)
        y_test=scaler.inverse_transform(y_test)
        return X_train, X_test, y_train, y_test, train_predict, test_predict

    def CrossValScore(self,train_predict, y_train, test_predict, y_test):
        #Get r2 score of the regression prediction
        return r2_score(train_predict,y_train), r2_score(test_predict,y_test)

    def CombineResults(self,data,X_train, X_test, y_train, y_test, train_predict, test_predict):
        #Combine the predictions to the dataframe
        y_rec = np.concatenate([y_train,y_test],axis = 0)
        y_pred = np.concatenate([train_predict , test_predict], axis= 0)
        X_rec = np.concatenate([X_train,X_test],axis = 0)
        df = pd.DataFrame(index = data.iloc[:-1,:].index)
        df["Close"] = X_rec
        df["Pred"] = y_pred
        df["Target"] = y_rec
        return df

    def __init__(self,data_path, output_scaler_path, output_model_path, output_data_path):
        data = joblib.load(data_path)
        X = data.drop(["Target"],1).iloc[:-1,:]
        y = data["Target"][:-1] 
        X_train, X_test, y_train, y_test = self.DataSplit(X,y)
        scaler=MinMaxScaler(feature_range=(0,1))
        X_train, X_test, y_train, y_test = self.Fit_Scaler(X_train, X_test, y_train, y_test, scaler)
        model = self.create_model(X_train.shape[0])      
        model, train_pred, test_pred = self.train_validate_predict(X_train, X_test, y_train, y_test, model)
        logging.info("Model traning complete")
        X_train, X_test, y_train, y_test, train_pred, test_pred = self.ScalerInvert(scaler, X_train, X_test, y_train, y_test, train_pred, test_pred)
        train_score, test_score = self.CrossValScore(train_pred, y_train, test_pred, y_test)
        results = self.CombineResults(data, X_train, X_test, y_train, y_test, train_pred, test_pred)
        model.save_weights(output_model_path)
        joblib.dump(scaler, output_scaler_path)
        joblib.dump(results, output_data_path)
        self.results = results
        self.train_score = train_score
        self.test_score = test_score
        self.model = model
        self.scaler = scaler

if __name__ == "__main__":
    f = open('config/params.json')
    params = json.load(f)

    data_path = params["train"]["data_path"]
    output_scaler_path = params["train"]["output_scaler_path"]
    output_model_path = params["train"]["output_model_path"]
    output_data_path = params["train"]["output_data_path"]

    train = Training(data_path, output_scaler_path, output_model_path, output_data_path)
    results, train_score, test_score = train.results, train.train_score, train.test_score
    print("LSTM model training score = {}".format(train_score))
    print("LSTM model test score = {}".format(test_score))
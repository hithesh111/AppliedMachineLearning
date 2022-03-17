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
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

class Training():
    def DataSplit(self,X,y,train_size=0.8):
        """Split the input data into training and testing datasets
        Args:
            X (dataframe): Data with model attributes as it's columns
            y (array-like): Target column
            train_size (float): Fraction of datapoints to be used as training_data
                                In (0,1)
        
        Returns:
            X_train (dataframe): Training features
            X_test (dataframe): Test features
            y_train (array-like): Training target
            y_test (array-like): Test target
        """
        split = int(train_size*X.shape[0])
        X_train, X_test = X[:split], X.iloc[split:]
        y_train, y_test = y[:split], y[split:]
        return X_train, X_test, y_train, y_test

    def Fit_Scaler(self,X_train,X_test,y_train,y_test, scaler):
        """Fit a scaler to training attributes and then scale the rest using the same scaler
        Args:
            X_train (dataframe): Training features
            X_test (dataframe): Test features
            y_train (array-like): Training target
            y_test (array-like): Test target
            scaler : Scikit Learn MinMaxScaler instance 
        
        Returns:
            Scaled versions of:
                X_train (dataframe): Training features 
                X_test (dataframe): Test features
                y_train (array-like): Training target
                y_test (array-like): Test target
        """
        #Fit a scaler to training attributes and then scale the rest using same scaler
        X_train=scaler.fit_transform(np.array(X_train)).reshape(-1,1)
        X_test=scaler.transform(np.array(X_test)).reshape(-1,1)
        y_train=scaler.transform(np.array(y_train).reshape(-1,1))
        y_test=scaler.transform(np.array(y_test).reshape(-1,1))
        return X_train, X_test, y_train, y_test

    def create_model(self,input_length):
        """Initialize an LSTM model
        Args:
            input_length (integer): The length of the input data
        
        Returns:
            LSTM model compiled to optimize for mean squared error using Adam
        """
        #Create an empty LSTM model
        model=Sequential()
        model.add(LSTM(100,return_sequences=True,input_shape=(input_length,1)))
        model.add(LSTM(100, return_sequences = True))
        model.add(LSTM(100))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')
        return model

    def train_validate_predict(self,X_train,X_test,y_train,y_test, model):
        """Train the model using training data and predict on train and test features
        Args:
            X_train (dataframe): Training features
            X_test (dataframe): Test features
            y_train (array-like): Training target
            y_test (array-like): Test target
            model : LSTM model
        
        Returns:
            model : fitted model
            train_predict (array-like): Model predictions on training data
            test_predict (array-like): Model predictions on testing data
        """
        model.fit(X_train, y_train, epochs=50,validation_data=(X_test,y_test), verbose=1)
        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)
        return model, train_predict, test_predict

    def ScalerInvert(self,scaler, X_train,X_test, y_train, y_test, train_predict, test_predict):
        """Invert the scaling in training and testing data
        Args:
            scaler : fitted instance of MinMaxScaler
            X_train (dataframe): Scaled Training features
            X_test (dataframe): Scaled Test features
            y_train (array-like): Scaled Training target
            y_test (array-like): Scaled Test target
            train_predict (array-like): Model predictions on training data
            test_predict (array-like): Model predictions on testing data       
        Returns:
            Descaled versions of:
                X_train (dataframe): Training features 
                X_test (dataframe): Test features
                y_train (array-like): Training target
                y_test (array-like): Test target
                train_predict (array-like): Model predictions on training data
                test_predict (array-like): Model predictions on testing data
        """
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)
        X_train=scaler.inverse_transform(X_train)
        X_test=scaler.inverse_transform(X_test)
        y_train=scaler.inverse_transform(y_train)
        y_test=scaler.inverse_transform(y_test)
        return X_train, X_test, y_train, y_test, train_predict, test_predict

    def CrossValScore(self,train_predict, y_train, test_predict, y_test):
        """ Finds the r2 score of training and testing predictions
        Args:
            train_predict (array-like): Model predictions on training data
            y_train (array-like): Training target
            test_predict (array-like): Model predictions on test data
            y_test (array-like): Test target
        
        Returns:
            R^2 metric of prediction on training set (float)
            R^2 metric of prediction on test set (float)
        """
        return r2_score(train_predict,y_train), r2_score(test_predict,y_test)

    def CombineResults(self,data,X_train, X_test, y_train, y_test, train_predict, test_predict):
        """ Collect prediction columns along with the input data
        Args:
            data (dataframe): Input dataframe
            X_train (dataframe): Training features
            X_test (dataframe): Test features
            y_train (array-like): training target
            y_test (array-like): test target
            train_predict (array-like): Model predictions on training data
            test_predict (array-like): Model predictions on test data
        
        Returns:
             Combined dataframe containing initial data as well as the predictions
        """
        y_rec = np.concatenate([y_train,y_test],axis = 0)
        y_pred = np.concatenate([train_predict , test_predict], axis= 0)
        X_rec = np.concatenate([X_train,X_test],axis = 0)
        df = pd.DataFrame(index = data.iloc[:-1,:].index)
        df["Close"] = X_rec
        df["Pred"] = y_pred
        df["Target"] = y_rec
        return df

    def __init__(self,data_path, output_scaler_path, output_model_path, output_data_path):
        """ init function of Training class
            Takes in input dataframe, splits the data, scales it, trains a model,
            descales it, gets predictions, saves scaler, output data and model.
        Args:
            data_path (string): Path of input data
            output_scaler_path(string) : location to store fitted scaler
            output_model_path (string) : location to store the fitted LSTM model
            output_data_path (string) : location to store combined data

        """
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
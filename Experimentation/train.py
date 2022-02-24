import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = joblib.load("Experimentation/joblib/data.joblib")
X = data.drop(["Target"],axis=1).iloc[200:-31,:]
y = data["Target"].iloc[200:-31]

split = int(X.shape[0]*0.8)
X_train, X_test, y_train, y_test = X.iloc[:split,:], X.iloc[split:,:], y.iloc[:split], y.iloc[split:]

attributes = X.columns

#Predict Majority Class
majority = np.zeros((X_test.shape[0])) + int(y.mean())
score = accuracy_score(y_test,majority)
print("\nMajority Prediction:",score)

LR = LogisticRegression(solver="liblinear", penalty = "l1")
# LR = RandomForestClassifier()
LR.fit(X_train[attributes],y_train)
pred = LR.predict(X_test[attributes])
pred_train = LR.predict(X_train[attributes])
train_score = accuracy_score(y_train,pred_train)
test_score = accuracy_score(y_test,pred)
print("\nLogistic Regression Train Score:",train_score)
print("\nLogistic Regression Test Score:",test_score)

X_test["TomPrediction"] = pred
X_test["TodayPrediction"] = X_test["TomPrediction"].shift(1)

taken_trades = X_test[X_test["TodayPrediction"]==1]
print(taken_trades["Close-Open"].sum())

joblib.dump(LR,"Experimentation/joblib/model.joblib")

print("\n")
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

data = joblib.load("Experimentation/data.joblib")
X = data.drop(["Target"],axis=1).iloc[200:-1,:]
y = data["Target"].iloc[200:-1]

split = int(X.shape[0]*0.8)
X_train, X_test, y_train, y_test = X.iloc[:split,:], X.iloc[split:,:], y.iloc[:split], y.iloc[split:]

attributes = X.columns
remove_attributes = ["Volume"]

attributes = [x for x in attributes if "Volume" not in x]
# attributes = [x for x in attributes if "Open" not in x]
attributes = [x for x in attributes if "Close" not in x]
attributes = [x for x in attributes if "High" not in x]
attributes = [x for x in attributes if "Low" not in x]
attributes = [x for x in attributes if "PrevTarget" not in x]

print(data[attributes].sample(5))

#Predict Majority Class
majority = np.zeros((X_test.shape[0])) + int(y.mean())
score = accuracy_score(y_test,majority)
print("\nMajority Prediction:",score)

LR = LogisticRegression(solver="liblinear", penalty = "l1")
LR = XGBClassifier()
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

print("\n")
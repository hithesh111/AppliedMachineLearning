import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

data = joblib.load("Experimentation/CAPLIN_POINT.joblib")
X = data.drop(["Target"],axis=1).iloc[4:-1,:]
y = data["Target"].iloc[4:-1]

split = int(X.shape[0]*0.8)
test_size = 100
X_train, X_test, y_train, y_test = X.iloc[:split,:], X.iloc[split:,:], y.iloc[:split], y.iloc[split:]
# X_train, X_test, y_train, y_test = X.iloc[:-test_size,:], X.iloc[-test_size:,:], y.iloc[:-test_size], y.iloc[-test_size:]

attributes = ["Volatility", "Percent_Change", "Open"]

#Predict Majority Class
majority = np.zeros((X.shape[0])) + int(y.mean())
score = accuracy_score(y,majority)
print("\nMajority Prediction:",score)

KNN = KNeighborsClassifier(30)
KNN.fit(X_train[attributes],y_train)
pred = KNN.predict(X_test[attributes])
score = accuracy_score(y_test,pred)
print("\nKNN Score:",score)

LR = LogisticRegression()
LR.fit(X_train[attributes],y_train)
pred = LR.predict(X_test[attributes])
test_score = accuracy_score(y_test,pred)
print("\nLogistic Regression Test Score:",test_score)

SVM = SVC()
SVM.fit(X_train[attributes],y_train)
pred = SVM.predict(X_test[attributes])
score = accuracy_score(y_test,pred)
print("\nSVM Score:",score)

GNB = GaussianNB(priors=[1-y.mean(), y.mean()])
GNB.fit(X_train[attributes],y_train)
pred = GNB.predict(X_test[attributes])
score = accuracy_score(y_test,pred)
print("\nGNB Score:",score)


print("\n")

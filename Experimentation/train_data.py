import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

data = joblib.load("Experimentation/DABUR.joblib")
X = data.drop(["Target"],axis=1).iloc[30:-1,:]
y = data["Target"].iloc[30:-1]

split = int(X.shape[0]*0.8)
# test_size = 100
X_train, X_test, y_train, y_test = X.iloc[:split,:], X.iloc[split:,:], y.iloc[:split], y.iloc[split:]
# X_train, X_test, y_train, y_test = X.iloc[:-test_size,:], X.iloc[-test_size:,:], y.iloc[:-test_size], y.iloc[-test_size:]

attributes = X.columns

#Predict Majority Class
majority = np.zeros((X.shape[0])) + int(y.mean())
score = accuracy_score(y,majority)
print("\nMajority Prediction:",score)

LR = LogisticRegression(penalty = "l1", solver = "liblinear")
LR.fit(X_train[attributes],y_train)
pred = LR.predict(X_test[attributes])
pred_train = LR.predict(X_train[attributes])
train_score = accuracy_score(y_train,pred_train)
test_score = accuracy_score(y_test,pred)
print("\nLogistic Regression Train Score:",train_score)
print("\nLogistic Regression Test Score:",test_score)
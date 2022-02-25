import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

data = joblib.load("Experimentation/joblib/data.joblib")
X = data.drop(["Target"],axis=1).iloc[200:-200,:]
y = data["Target"].iloc[200:-200]

# data = data.sample(data.shape[0],replace=False, random_state=42)
# split = int(X.shape[0]*0.7)
# X_train, X_test, y_train, y_test = X.iloc[:split,:], X.iloc[split:,:], y.iloc[:split], y.iloc[split:]

attributes = list(X.columns)
attributes.remove("Close-Open")
# print("\n",attributes)

# model = LogisticRegression()
# params = {'tol':[1e-6,1e-5,1e-6,1e-4,1e-3,1e-2,1,10], 'C':[1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000]}

# model = RandomForestClassifier()
# params = {'n_estimators': list(range(50,200,10)),'max_depth': list(range(2,12)), 'min_samples_split':list(range(2,200,5)), 'min_samples_leaf': list(range(1,300,5))}

# model = GradientBoostingClassifier(max_depth = 1)
# model = GaussianNB()

# model = SVC()
# params = {'C':[1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000], 'degree':list(range(2,4))}

model = XGBClassifier()
params = {
            'eta': [x/100 for x in list(range(1,20))],
            'n_estimators': list(range(50,200,10)),
            'max_depth': list(range(2,12)),
            'max_leaf_nodes':list(range(2,200,5)),
            'min_child_weight': list(range(1,300,5)),
            'colsample_bytree' : [x/10 for x in list(range(5,11))],
            'scale_pos_weight': [float(x)/10 for x in range(1,21)]
            }

GS = RandomizedSearchCV(model,params,scoring = 'precision', n_iter=50)
model = GS

model.fit(X[attributes],y)

print(model.best_params_)
print(model.best_score_)

#Predict All 1's
allones = np.zeros((X.shape[0])) + 1
score = precision_score(y,allones)
print("\nTradeEveryday Prediction:",score)

#Predict opposite of today
opposite = X["Close-Open"].apply(lambda x: 0 if x>0 else 1)
score = precision_score(y, opposite)
print("\nOpposite Prediction", score)

#Predict same as today
same = X["Close-Open"].apply(lambda x: 1 if x>0 else 0)
score = precision_score(y, same)
print("\nSame Prediction", score)

joblib.dump(model,"Experimentation/joblib/model.joblib")

print("\n")
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score

df = joblib.load("Experimentation/joblib/data.joblib").iloc[-200:-1,:]

unseen_data = df.drop(["Close-Open"],axis = 1)

unseen_X = unseen_data.drop(["Target"],axis = 1)
unseen_Y = unseen_data["Target"]

model = joblib.load("Experimentation/joblib/model.joblib")
predictions = model.predict(unseen_X)
df["pred"] = predictions
df["Profit"] = df["Close-Open"].shift(-1)*df["pred"]

print(df[df["pred"]==1])
print("Profit = ", round(df["Profit"].sum(),2))

score = precision_score(unseen_Y,predictions)
print("Precision",score)

joblib.dump(score,"Experimentation/joblib/scores.joblib")

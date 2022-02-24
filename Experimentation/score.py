import joblib
import numpy as np
from sklearn.metrics import accuracy_score

unseen_data = joblib.load("Experimentation/joblib/data.joblib").iloc[-31:-1,:]

unseen_X = unseen_data.drop(["Target"],axis = 1)
unseen_Y = unseen_data["Target"]

model = joblib.load("Experimentation/joblib/model.joblib")
predictions = model.predict(unseen_X)

probs = model.predict_proba(unseen_X)

score = accuracy_score(predictions, unseen_Y)

joblib.dump(score,"Experimentation/joblib/scores.joblib")

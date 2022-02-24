from sklearn.metrics import confusion_matrix
import joblib

unseen_data = joblib.load("Experimentation/joblib/data.joblib").iloc[-31:-1,:]

unseen_X = unseen_data.drop(["Target"],axis = 1)
unseen_Y = unseen_data["Target"]

model = joblib.load("Experimentation/joblib/model.joblib")
predictions = model.predict(unseen_X)

probs = model.predict_proba(unseen_X)

conf_mat = confusion_matrix(predictions, unseen_Y, normalize='true')

print(conf_mat)
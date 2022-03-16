import joblib
import matplotlib.pyplot as plt

data = joblib.load("joblib/unseen.joblib")
plt.plot(data["y"])
plt.plot(data["Pred"])
plt.legend(["Actual","Prediction"])
plt.savefig("plots/unseen_predictions.png")
plt.close()
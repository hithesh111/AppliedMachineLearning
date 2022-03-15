import joblib
import matplotlib.pyplot as plt
results = joblib.load("joblib/train_results.joblib")

plt.plot(results["Target"][-365:])
plt.plot(results["Pred"][-365:])
plt.savefig("plots/1yearpredictions.jpg")
plt.close()

plt.plot(results["Target"][-90:])
plt.plot(results["Pred"][-90:])
plt.savefig("plots/3monthspredictions.jpg")
plt.close()

plt.plot(results["Target"][-30:])
plt.plot(results["Pred"][-30:])
plt.savefig("plots/30dayspredictions.jpg")
plt.close()
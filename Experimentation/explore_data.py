import joblib
import matplotlib.pyplot as plt
data = joblib.load("Experimentation/data.joblib")

plt.plot(data["Close"])

plt.savefig('Experimentation/plots/ClosePrices.png')
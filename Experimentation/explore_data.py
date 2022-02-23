import joblib
import matplotlib.pyplot as plt
data = joblib.load("Experimentation/data.joblib")

data[["Close","SMA50","SMA100","SMA200"]].plot()
plt.savefig('Experimentation/plots/MovingAverages.png')

plt.close()
data[["Close","SMM50","SMM100","SMM200","SMm50","SMm100","SMm200"]].plot()
plt.savefig('Experimentation/plots/MovingMaximum.png')

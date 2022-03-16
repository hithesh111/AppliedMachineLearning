import joblib
import json
import matplotlib.pyplot as plt

class Visualization():
    def load_data(self, score_data_path):
        data = joblib.load(score_data_path)
        return data
    
    def plot(self,data):
        plt.plot(data["y"])
        plt.plot(data["Pred"])
        plt.legend(["Actual","Prediction"])
        plt.savefig("plots/unseen_predictions.png")
        plt.close()

    def __init__(self, score_data_path):
        data = self.load_data(score_data_path)
        self.plot(data)
    
if __name__ == "__main__":
    f = open('config/params.json')
    params = json.load(f)

    score_data_path = params["explore_data"]["score_data_path"]
    Visualization(score_data_path)
from sklearn.metrics import r2_score, precision_score, recall_score, f1_score
import joblib
import json

class Evaluate():
    def load(self, unseen_data_path):
        unseen_data = joblib.load(unseen_data_path)
        return unseen_data

    def get_pred_for_tomorrow(self, unseen_data):
        today_date = unseen_data.index[-1]  
        tom_pred = unseen_data.loc[unseen_data.index[-1],"Pred"]
        return today_date, tom_pred

    def add_binary_target(self, unseen_data):
        unseen_data["Target"] = (unseen_data["y"] - unseen_data["OpenTom"]).apply(lambda x: 1 if x>0 else 0)
        unseen_data["Target_Pred"] = (unseen_data["Pred"] - unseen_data["OpenTom"]).apply(lambda x: 1 if x>0 else 0)
        return unseen_data
    
    def error_metrics(self, unseen_data):
        r2score = r2_score(unseen_data["y"],unseen_data["Pred"])
        precision = precision_score(unseen_data["Target"],unseen_data["Target_Pred"])
        recall = recall_score(unseen_data["Target"],unseen_data["Target_Pred"])
        f1score =  f1_score(unseen_data["Target"],unseen_data["Target_Pred"])
        return r2score, precision, recall, f1score

    def total_profit(self, unseen_data):
        taken_trades = unseen_data[unseen_data["Target_Pred"]==1]
        self.n_trades = taken_trades.shape[0]
        profit = (taken_trades["Pred"] - taken_trades["OpenTom"]).sum()
        return profit

    def __init__(self, unseen_data_path):
        unseen_data = self.load(unseen_data_path)
        self.min_price = unseen_data["Close"].min()
        self.mean_price = unseen_data["Close"].mean()
        self.max_price = unseen_data["Close"].max()
        self.n_days = unseen_data.shape[0]
        self.today_date, self.tom_pred = self.get_pred_for_tomorrow(unseen_data)
        unseen_data = self.add_binary_target(unseen_data)
        self.r2score, self.precision, self.recall, self.f1score = self.error_metrics(unseen_data)
        self.profit = self.total_profit(unseen_data)

if __name__ == "__main__":
    f = open('config/params.json', )
    params = json.load(f)

    unseen_data_path = params["evaluate"]["unseen_data_path"]
    evaluate = Evaluate(unseen_data_path)
    
    print("r2_score = {}".format(evaluate.r2score))
    print("precision = {}".format(evaluate.precision))
    print("recall = {}".format(evaluate.recall))
    print("f1_score = {}".format(evaluate.f1score))
    print("\n")
    print("Mean Stock Price = {}".format(evaluate.mean_price))
    print("Total profit over {} days = {}".format(evaluate.n_days,round(evaluate.profit),2))
    print("Approx % profit over {} days = {}".format(evaluate.n_days, round(100*evaluate.profit/evaluate.mean_price,2)))
    print("Approx Projected Annual profits = {}".format(250*evaluate.profit/evaluate.n_days))
    print("Prediction for Tomorrow = {} as on {}".format(evaluate.tom_pred, evaluate.today_date))
    
    



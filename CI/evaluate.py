from sklearn.metrics import r2_score, precision_score, recall_score, f1_score
import joblib

unseen_data = joblib.load('joblib/unseen.joblib')

today_date = unseen_data.index[-1]  
tom_pred = unseen_data.loc[unseen_data.index[-1],"Pred"]

percent_one = unseen_data.loc[unseen_data.index[-1],"Close"]/200
unseen_data["Target"] = (unseen_data["y"] - unseen_data["OpenTom"]).apply(lambda x: 1 if x>0 else 0)
unseen_data["Target_Pred"] = (unseen_data["Pred"] - unseen_data["OpenTom"]).apply(lambda x: 1 if x>0 else 0)

print("\n R^2:",r2_score(unseen_data["y"],unseen_data["Pred"]))
print("Precision:", precision_score(unseen_data["Target"],unseen_data["Target_Pred"]))
print("Recall:", recall_score(unseen_data["Target"],unseen_data["Target_Pred"]))
print("F1 Score:", f1_score(unseen_data["Target"],unseen_data["Target_Pred"]))

taken_trades = unseen_data[unseen_data["Target_Pred"]==1]
profit = (taken_trades["Pred"] - taken_trades["OpenTom"]).sum()
print("Profit Made:", profit)

profit_per_day = profit/unseen_data.shape[0]
print("Profit per Day:", profit_per_day)

daily_percentage_profit = 100*profit_per_day/(unseen_data.loc[unseen_data.index[0],"Open"])
print("Daily % profit:", daily_percentage_profit)

print(unseen_data.tail())

print("Prediction for Tomorrow = {}".format(tom_pred), "on ", today_date)

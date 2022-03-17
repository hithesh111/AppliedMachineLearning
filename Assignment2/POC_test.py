import json
from prepare_data import Prepare
from train import Training
from score import Score
from evaluate import Evaluate, Benchmark
import numpy as np

f = open('config/params.json')
params = json.load(f)

stock_ticker = params["prepare_data"]["stock_ticker"]
start_date = params["prepare_data"]["start_date"]
end_date = params["prepare_data"]["end_date"]
output_data_path = params["prepare_data"]["data_path"]

input_data_path = params["train"]["data_path"]
output_scaler_path = params["train"]["output_scaler_path"]
output_model_path = params["train"]["output_model_path"]
output_data_path_train = params["train"]["output_data_path"]

stock_ticker = params["score"]["stock_ticker"]
start_date_score = params["score"]["start_date"]
model_weights_path = params["score"]["model_weights_path"]
input_scaler_path = params["score"]["scaler_path"]
output_data_path_score = params["score"]["output_data_path"]

unseen_data_path = params["evaluate"]["unseen_data_path"]

profits = []
Prepare(stock_ticker, start_date, end_date, output_data_path)
n_iterations = 50
for i in range(n_iterations):
    Training(input_data_path, output_scaler_path, output_model_path, output_data_path_train)
    Score(stock_ticker, start_date_score, model_weights_path, input_scaler_path, output_data_path_score)
    evaluate = Evaluate(unseen_data_path)
    profit = evaluate.profit
    profits.append(profit)
no_days = evaluate.n_days

print("\n")
bm = Benchmark(unseen_data_path)
print("Trade everyday Benchmark profit = {}".format(bm.benchmark1_profit))
print("Number of iterations:", n_iterations)
print("Percentage of iterations in which a profit was taken: ", 100*sum(np.array(profits)>0)/n_iterations)
print("Mean Stock Price: ", evaluate.mean_price)
print("Average Profit: ", np.mean(profits), "in a period of ", no_days, " days.")
print("Projected Annual Profit Per Stock: ", np.mean(profits)*250/no_days)
print("\n")
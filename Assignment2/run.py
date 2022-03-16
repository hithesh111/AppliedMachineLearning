import json
from prepare_data import Prepare
from train import Training
from score import Score
from evaluate import Evaluate
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

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

Prepare(stock_ticker, start_date, end_date, output_data_path)
Training(input_data_path, output_scaler_path, output_model_path, output_data_path_train)
Score(stock_ticker, start_date_score, model_weights_path, input_scaler_path, output_data_path_score)
evaluate = Evaluate(unseen_data_path)

print("\n")
print("Profit Made per stock: ", evaluate.mean_price)
print("Profit Made per stock: ", evaluate.profit)
print("\n")
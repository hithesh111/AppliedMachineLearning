# AML Assignment 2

## Added functionalities compared to Assignment 1
1. Modular Design
2. Parameter Config
3. Logging
4. Comparison with Benchmaks
5. Proof Of Concept Simulation

## Context
We have daily stock price data available from Yahoo Finance. We would like to build an ML model to make use of the data to make predict info about future prices and hope to make profitable buy/sell orders.

Data Source: https://analyzingalpha.com/yfinance-python

## Objective
Make profitable trades if Closing Price is greater than Open Price.
Use this information to buy at Open and Sell at Close.

## Success metrics
1. Amount of profit during a period of Days

## ML Approach
Building an LSTM to predict tomorrows' closing price from OHLC data from Yahoo Finance and using the prediction to decide whether to take a trade at Open tomorrow if predicted Close is greater than tomorrow's Open.

Key assumptions: No Broker Charges, Buy exactly at Open, Sell exactly at Close, No short selling

## Performance Metrics
1. R2 Score
2. Precision
3. Recall
4. F1 Score

## Features
1. Today's Close

## Target
1. Tomorrow's Close

## Benchmarks
1. Take a trade everyday
2. Predict same outcome as today
3. Predict opposite outcome as today
4. Predict moving average
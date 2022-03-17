# AML Assignment 2

## Changes compared to Assignment 1
1. Modular Design
2. Parameter Config
3. Logging
    * Ideally, we would want to send it to specified locations and make use of custom formatting. But here, it only emits log messages on the console.
4. Comparison with Benchmark
5. Proof Of Concept Simulation
6. Documentation

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

### LSTM

* LSTM uses feedback connections.
* Used for time series because of uncertainty in which lags could be important.
* Gating mechanism of LSTM takes care of which values to store and which to forget.
* Deals with vanishing gradients problem compared to RNNs
* Still very prone to exploding gradients problem
* Therefore feature scaling is very necessary

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

# AML Assignment

## Context
We have daily stock price data available from Yahoo Finance. We would like to build an ML model to make use of the data to make predict info about future prices and hope to make profitable buy/sell orders.

Data Source: https://analyzingalpha.com/yfinance-python

## Objective
Make profitable trades if Closing Price is greater than Open Price.
Use this information to buy at Open and Sell at Close.

## Success metrics
1. Percentage profit over a week
2. Percent of Trades Won (100*No of Actual Wins/Total No of Predicted Wins)

## ML Approach
Building a binary classifier from OHLC data from Yahoo Finance to predict if tomorrow's Close would be greater tomorrow's Open price

## Performance Metrics
1. Accuracy
2. Precision
3. Recall

## Features
1. Previous Day's OHLC
2. Previous Day's Volume
3. Previous Day's [Close - Open]

## Target
1. Is tomorrow's Close > Open

## Benchmarks
1. Predict yes on everyday
2. Predict previous day's target value
3. Predict opposite of previous day's target value
4. Logistic Regression on [Open - Close] or Open, Close, [Open - Close]
5. Gaussian Naive Bayes Model on all features
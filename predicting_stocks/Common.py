import json
import os

import matplotlib.pyplot as plt
import ast
import pandas as pd
from yahoofinancials import YahooFinancials
from tensorflow.keras.models import load_model
from Trading.data_order_something import read_data
import yfinance as yf

X_VALUES = [['open', 'low', 'high', 'close'], ['Open', 'Low', 'High', 'Close']]


def write_in_file(path, data):
    with open(path, 'a') as file:
        file.write(data)
        file.close()


def read_csv(path, ticker=None, other='3'):
    return pd.read_csv(path) if os.path.exists(path) else pd.read_csv(read_data(ticker, other=other))


def iterate_data(data, what=0):
    return [[data[key][index]
             for key in X_VALUES[what]]
            for index, i in enumerate(data['close'] if 'close' in data.keys() else data['Close'])]


def load_model_from_file(ticker, ):
    """
    Function to load model from saved model file
    """

    return load_model(path) if os.path.exists((path := f'saved_model/{ticker}_model')) else None


def get_historical_data(ticker, start, end):
    return (pd.DataFrame(
        YahooFinancials(ticker).get_historical_price_data(
            start_date=start,
            end_date=end,
            time_interval='daily')[
            ticker]['prices']).drop('date', axis=1).set_index('formatted_date'))


def plot(data, pre_prices, ticker):
    plt.plot(data, color='blue')
    plt.plot(pre_prices, color='red')
    plt.title(f'{ticker} Share Price')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Share Price')
    plt.legend()
    plt.show()


def write_in_json_file(path, data, ticker=None):
    with open(path, "r") as a_file:
        data = json.dumps(ast.literal_eval(data))
        print(data)
        json_object = json.load(a_file)
        json_object[ticker] = ast.literal_eval(data)[ticker]
        a_file = open(path, "w")
        json.dump(json_object, a_file)


def return_json_data(ticker, json_path='../predicting_stocks/settings_for_ai/parameters_status.json'):
    try:
        with open(json_path, 'r'):
            print(json_path)
    except FileNotFoundError:
        print('did not find file', json_path, sep=', ')
        return [None, None, None, None]

    if not json_path:
        return None
    with open(json_path, 'r') as json_file:
        p = json.load(json_file)
        if ticker in p:
            p = p[ticker]['settings']
            return [p['epochs'], p['units'], p['prediction_days'], p['prediction_day']]
        else:
            return [None, None, None, None]


# def save_historical_data(ticker, start=START, end=END):
#     """
#
#     :param ticker: Ticker to save the historical data
#     :param start: from what date (if nothing so the big START date)
#     :param end:  'til what date  (if nothing so the big END date which it is today by default)
#
#     :Doing saving an historical data of a stock into file in /Data/ticker.txt
#     *IMPORTANT* this func must use internet to be used otherwise you can't get the data to save
#     """
#     data = get_historical_data(ticker, start, end)
#     data = {ticker: dict((i, list(data[i].values)) for i in data)}
#     with open(f'./Data/{ticker}.txt', 'w') as t:
#         t.write(str(data))
#         t.close()

def check_data(x_train, y_train, constant=1):
    for i in range(0, len(x_train) - constant):
        if y_train[i] != x_train[i + constant][-1]:
            raise InterruptedError("something went wrong in the code please check it")


def get_data_from_saved_file(ticker, ):
    """

    :param ticker: Ticker to get historical data from its file
    :return: dictionary of historical data of a stock that has been saved earlier using save_historical_data()

    Format - ticker name.txt in Data directory, otherwise it will not find the data
    """
    file = open(f'./Data/{ticker}.txt', 'r')
    data = file.read()
    return ast.literal_eval(data)


def read_from_file(ticker, other):
    try:
        with open(f'../Trading/Historical_data/{ticker} - {other}.txt', 'r') as file:
            return file.read()

    except FileNotFoundError:
        print('file not found')
        return None


def get_data_from_file_or_yahoo(ticker, other):
    return iterate_data(ast.literal_eval(data), what=1) if (data := read_from_file(ticker, other)) is not None else \
        intraday_with_yahoo(ticker, other)


def intraday_with_yahoo(ticker, other: [str, int] = '3'):
    data = yf.download(tickers=ticker, period=f'{str(other)}d', interval='1m')
    with open(f'../Trading/Historical_data/{ticker} - {other}.txt', 'w') as file:
        data_dict = dict((key, [i for i in data[key]]) for key in ['Open', 'Close', 'Low', 'High'])
        file.write(str(data_dict))
    return iterate_data(data_dict, what=1)


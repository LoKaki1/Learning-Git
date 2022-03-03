import json
import os

import datetime as dt
import matplotlib.pyplot as plt
import ast
import pandas as pd
from yahoofinancials import YahooFinancials
from tensorflow.keras.models import load_model
from Trading.data_order_something import read_data, read_from_file
import yfinance as yf
import re

import time

X_VALUES = [['open', 'low', 'high', 'close'], ['Open', 'Low', 'High', 'Close']]


def write_in_file(path, data):
    with open(path, 'a') as file:
        file.write(data)
        file.close()


def str_time_prop(start, end, time_format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formatted in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, time_format))
    etime = time.mktime(time.strptime(end, time_format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(time_format, time.localtime(ptime))


def random_date(start, end, prop):
    return str_time_prop(start, end, '%m/%d/%Y %I:%M %p', prop)


def read_csv(path, ticker=None, other='3'):
    return pd.read_csv(path) if os.path.exists(path) else pd.read_csv(read_data(ticker, other=other))


def iterate_data(data, what=0):
    return [[float(p)
             for key in X_VALUES[what] if re.match('^[0-9/.]*$', str(p := data[key][index])) is not None]
            for index, i in enumerate(data['close'] if 'close' in data.keys() else data['Close'])]


def try_except(catch_except, func, *args):
    try:
        func(args[-1])
        return True
    except catch_except:
        return False


def load_model_from_file(self):
    """
    Function to load model from saved model file
    """

    return load_model(
        path) if os.path.exists(
        (path := f'saved_model/'
                 f'{self.ticker}_model/'
                 f'{self.epochs}-'
                 f'{self.units}-'
                 f'{self.prediction_days}-'
                 f'{self.prediction_day}')) else None


def get_historical_data(ticker, start, end):
    data = (pd.DataFrame(
        YahooFinancials(ticker).get_historical_price_data(
            start_date=start,
            end_date=end,
            time_interval='daily')[
            ticker]['prices']).drop('date', axis=1)
            .set_index('formatted_date'))

    print(data['close'][-1])
    return data


def plot(data, pre_prices, ticker):
    plt.plot(data, color='blue')
    plt.plot(pre_prices, color='red')
    plt.title(f'{ticker} Share Price')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Share Price')
    plt.legend()
    plt.show()


def write_in_json_file(path, data: dict, ticker=None):
    with open(path, 'r') as read_file:
        data = json.dumps(data)
        json_object = json.load(read_file)
        json_object[ticker] = ast.literal_eval(data)[ticker]
        with open(path, 'w') as write_file:
            json.dump(json_object, write_file)


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
        json_file.close()
        if ticker in p:
            p = p[ticker]['settings']
            return [p['epochs'], p['units'], p['prediction_days'], p['prediction_day']]
        else:
            return [None, None, None, None]


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


def get_data_from_file_or_yahoo(ticker):
    return iterate_data(ast.literal_eval(data), what=1) if (data := read_from_file(ticker)) is not None else \
        intraday_with_yahoo(ticker)


def intraday_with_yahoo(ticker, other: [str, int] = '3'):
    data = yf.download(tickers=ticker, period=f'{str(other)}d', interval='1m')
    with open(f'../Trading/Historical_data/{ticker}.txt', 'w') as file:
        data_dict = dict((key, [i for i in data[key]]) for key in ['Open', 'Close', 'Low', 'High'])
        file.write(str(data_dict))
        file.close()
    return iterate_data(data_dict, what=1)


def open_json_file(path):
    with open(path, 'r') as file:
        return json.loads(file.read())


def handle_with_time(ticker, json_object):
    today_str = dt.datetime.now().strftime('%d-%b-%Y')
    ticker_last_date = json_object
    if ticker not in ticker_last_date:
        return today_str
    ticker_last_date = ticker_last_date[ticker]
    return today_str if dt.datetime.strptime(
        ticker_last_date[
            'date'], '%d-%b-%Y') < dt.datetime.strptime(today_str, '%d-%b-%Y') else float(ticker_last_date['price'])


def get_last_id(json_object):
    return len(json_object) + 1


def save_in_data_base(ticker, price, settings, date, _id):
    data = {
        ticker: {
            "id": _id,
            "price": price,
            "settings": settings,
            "date": date,
            "current_price": (t := str(get_last_price(ticker)))[0:6 if len(t) >= 6 else -1]
        }
    }
    write_in_json_file('database.json', data, ticker)


def get_last_price(ticker):
    return get_historical_data(ticker, (t := (dt.datetime.now()) - dt.timedelta(days=1)).strftime('%Y-%m-%d'),
                               dt.datetime.now().strftime('%Y-%m-%d'),)['close'][-1]


def open_json(path):
    with open(path, 'r') as file:
        json_data = json.loads(file.read())
    return json_data

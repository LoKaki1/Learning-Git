import json
import matplotlib.pyplot as plt
import ast


def write_in_file(path, data):
    with open(path, 'a') as file:
        file.write(data)
        file.close()


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


def return_json_data(ticker, json_path=r'../predicting_stocks/settings_for_ai/parameters_status'):
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


def save_historical_data(ticker, start=START, end=END):
    """

    :param ticker: Ticker to save the historical data
    :param start: from what date (if nothing so the big START date)
    :param end:  'til what date  (if nothing so the big END date which it is today by default)

    :Doing saving an historical data of a stock into file in /Data/ticker.txt
    *IMPORTANT* this func must use internet to be used otherwise you can't get the data to save
    """
    data = get_historical_data(ticker, start, end)
    data = {ticker: dict((i, list(data[i].values)) for i in data)}
    with open(f'./Data/{ticker}.txt', 'w') as t:
        t.write(str(data))
        t.close()


def get_data_from_saved_file(ticker, ):
    """

    :param ticker: Ticker to get historical data from its file
    :return: dictionary of historical data of a stock that has been saved earlier using save_historical_data()

    Format - ticker name.txt in Data directory, otherwise it will not find the data
    """
    file = open(f'./Data/{ticker}.txt', 'r')
    data = file.read()
    return ast.literal_eval(data)


return_json_data('NIO')
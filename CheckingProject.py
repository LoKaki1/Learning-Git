from yahoofinancials import YahooFinancials
import datetime as dt
import pandas as pd

test_start = dt.datetime(2021, 8, 27).strftime('%Y-%m-%d')
test_end = (dt.datetime.now() - dt.timedelta(days=0)).strftime('%Y-%m-%d')

STOCKS_PRICE = {}


def get_historical_data(ticker, start=test_start, end=test_end):
    ticker = ticker.strip("'")
    data = YahooFinancials(ticker)
    print(ticker)
    data = data.get_historical_price_data(start, end, 'daily')

    t_data = pd.DataFrame(data[ticker]['prices'])
    t_data = t_data.drop('date', axis=1).set_index('formatted_date')
    t_data.head()

    return t_data


def read_from_file(path):
    with open(path, 'r') as file:
        data = file.read()
        file.close()
    return data


def check_data(data=read_from_file('Prediction.txt')):
    data = data.split('\n')
    print(data)
    ratio = 0
    for i in data:
        words = i.split(' ')
        if words.__len__() >= 8:
            t = ''.join([words[1], ' ', words[-1]])
            if t not in list(STOCKS_PRICE.keys()):
                STOCKS_PRICE[t] = real_value = float(get_historical_data(words[1], start=test_start, end=words[-1])['close'][-1])

            else:
                real_value = STOCKS_PRICE[t]
            predicted_price = float(words[8])
            ratio += min(real_value / predicted_price, predicted_price / real_value)
    return ratio/len(data)


print(check_data())

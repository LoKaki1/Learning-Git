import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import time
from fucck import predict_stocks
import json
import datetime as dt
from Finanace_something.Interactive import example_fetching_time_data
from whatsapp import write_in_file

BOUGHT_TICKERS = []
SHORT_TICKERS = []

api_key = 'RNZPXZ6Q9FEFMEHM'


def get_last_price(ticker):
    try:
        example_fetching_time_data.ever(ticker)
    except OSError as e:
        pass
    finally:
        pass
    file = open(r'prices.txt', 'r')
    data = (file.read()).split("\n")
    file.close()
    print(data)
    return float(data[-1])


def get_tickers_prediction(tickers):
    tickers_dict = {}
    for i in tickers:
        tickers_dict[i] = [predict_stocks([i], units='272', prediction_days='63', prediction_day='1')]
    return tickers_dict


def main():

    data = []
    tickers = [ 'LI', 'XPEV', 'MARA', "RIOT", 'UAA', 'CEI', 'NIO']
    prediction_tickers = {}
    for i in tickers:
        prediction_tickers[i] = [predict_stocks([i], units='272', prediction_day='1', prediction_days='63')]
    write_in_file( str(prediction_tickers), 'P.txt')
    money = 0
    last_price = 0
    predict_price = 0
    data = ""

    while True:
        for i in prediction_tickers.keys():
            predict_price = prediction_tickers[i][0]
            last_price = get_last_price(i)
            last_money = money
            # Buy if Lower then prediction

            if last_price < predict_price and i not in BOUGHT_TICKERS and i not in SHORT_TICKERS:
                data += ''.join(["\nBought ", i, " in price ", str(last_price)])
                print("Bought ", i, " in price ", last_price)
                prediction_tickers[i].append(last_price)
                BOUGHT_TICKERS.append(i)

            elif last_price >= predict_price and i in BOUGHT_TICKERS:
                print("Sold ", i, " in price ", last_price)
                prediction_tickers[i].append(last_price)
                BOUGHT_TICKERS.remove(i)
                money += prediction_tickers[i][-1] - prediction_tickers[i][-2]
                data += ''.join(["\n Sold ", i, " in price ", str(last_price)])

            elif last_price > predict_price and i not in SHORT_TICKERS and i not in BOUGHT_TICKERS:
                print("\n Shorted ", i, " in price ", last_price)
                prediction_tickers[i].append(last_price)
                SHORT_TICKERS.append(i)
                data += ''.join(["\n Shorted ", i, " in price ", str(last_price)])

            elif last_price <= predict_price and i in SHORT_TICKERS:
                print("Covered ", i, " i n price ", last_price)
                prediction_tickers[i].append(last_price)
                SHORT_TICKERS.remove(i)
                money += prediction_tickers[i][-2] - prediction_tickers[i][-1]
                data += ''.join(["\n Covered ", i, " in price ", str(last_price)])

            if data != '':
                write_in_file(data, 'Trades .txt')
            if last_money != money:
                write_in_file(''.join(['\n', str(money)]), 'money.txt')
                print('I\'ve got ', money)
            data = ''




if __name__ == '__main__':
    main()

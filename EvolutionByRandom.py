""""
This file will evolve the machine to be with the best settings for the market
"""
from StockPrediction import predict_stocks, get_historical_data
import os
from random import randint
from datetime import datetime, timedelta

NUMBER_RANGE = 9


def random_variables():
    epochs = randint(10, 30)
    batch_size = randint(5, 64)
    units = randint(19, 100)
    prediction_days = randint(10, 80)
    prediction_day = 1
    return [epochs, batch_size, units, prediction_days, prediction_day]


def date_minus_date_time(delta_time):
    return (datetime.now() - timedelta(delta_time)).strftime('%Y-%m-%d')


def predicting_value_minus_delta_times(ticker, EBUPP, delta_times):
    end_day = str(date_minus_date_time(delta_times))
    predicted_end_day = str(date_minus_date_time(delta_times+1))

    predict_value = predict_stocks([ticker], units=EBUPP[2], prediction_days=EBUPP[3], epochs=EBUPP[0],
                                   batch_size=EBUPP[1], prediction_day=EBUPP[4], end_day=predicted_end_day)
    real_value = get_historical_data(ticker, end=end_day)['close'][-1]
    print(predict_value, real_value, end_day)
    return predict_value, real_value


def get_ratio_from_all_days(ticker, EBUPP):
    real_value, predict_value = 0, 0
    huge_ratio = 0
    for i in range(NUMBER_RANGE):
        predict_value, real_value = predicting_value_minus_delta_times(ticker, EBUPP, i)
        ratio_for_this_delta = min(predict_value / real_value, real_value / predict_value)
        huge_ratio += ratio_for_this_delta
    return huge_ratio / NUMBER_RANGE, real_value, predict_value


def write_in_file(path='Generated_data.txt', data=None):
    file = open(path, 'a')
    file.write(data)
    file.close()


def get_the_best_of_them(ticker, best_ratio):

    for i in range(NUMBER_RANGE):
        variables_list = random_variables()
        ratio, real_value, predict_value = get_ratio_from_all_days(ticker, variables_list)
        if ratio > best_ratio:
            data = "".join(['\n', 'epochs=', str(variables_list[0]),
                            ' batch_size=', str(variables_list[1]),
                            ' units=', str(variables_list[2]),
                            ' prediction_days=', str(variables_list[3]),
                            ' prediction_day=', str(variables_list[4]),
                            ' ratio - ', str(ratio),
                            ' real value - ', str(real_value),
                            ' predict value - ', str(predict_value)[0:5],
                            ' number of days it checked on ', str(NUMBER_RANGE),
                            'ticker - ', str(ticker)
                            ])
            print(data)

            write_in_file(data=data)
            best_ratio = ratio

    return best_ratio


last_ratio = 0.98
while True:
    last_ratio = get_the_best_of_them('LI', last_ratio)
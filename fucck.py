import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_datareader.data as pdr
import datetime as dt
from whatsapp import str_file, write_in_file
from yahoofinancials import YahooFinancials
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# from Classfier_AI import get_historical_data

STUPID_IDEA = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.']
test_start = dt.datetime(2010, 1, 1).strftime('%Y-%m-%d')
test_end = (dt.datetime.now() - dt.timedelta(days=0)).strftime('%Y-%m-%d')
MY_X_AXIS = {'open': [], 'close': [], 'high': [], 'low': [], 'volume': [], 'adjclose': [], 'everything': []}
POSSIBLE = {'open': [], 'close': [], 'high': [], 'low': [], 'volume': [], 'adjclose': []}

EPOCHS = 25
BATCH_SIZE = 32
UNITS = 50
PREDICTION_DAYS = 21
PREDICTION_DAY = 1

What = 'close'

yf.pdr_override()


def my_pr(ticker, file='Prediction.txt'):
    data = str_file(file).split('\n')
    checked_data = []
    for i in data:
        if i.split(' - ')[1] == ticker:
            return analysis_prediction_data


def analysis_prediction_data(data):
    my_data = data.split(' - ')
    ticker = my_data[1]
    pp = my_data[5]
    date = my_data[7]
    return check_data(ticker, date, pp)


def check_data(ticker, date, price):
    data = get_historical_data(ticker, start=date, end=date)
    real_price = data['close'][-1]
    prediction_and_real_ratio = min(real_price / float(price), float(price) / real_price)
    return "\n".join(
        ["real price - ", str(real_price), "predict price - ", str(price), str(prediction_and_real_ratio)[0:5]])


def get_historical_data(ticker, start=test_start, end=test_end):
    ticker = ticker.strip("'")
    data = YahooFinancials(ticker)
    print(ticker)
    data = data.get_historical_price_data(start, end, 'daily')

    t_data = pd.DataFrame(data[ticker]['prices'])
    t_data = t_data.drop('date', axis=1).set_index('formatted_date')
    t_data.head()

    return t_data

def get_my_x_axis():
    t = {}
    for i in MY_X_AXIS:
        t[i] = []

    return t


def get_possible():
    t = {}
    for i in POSSIBLE:
        t[i] = []

    return t

def getting_stocks():
    file = open('../Trading/Data/stocks.txt', 'r')
    data = file.read()
    file.close()
    print(data)

    data = data.strip(']').strip('[').split(', ')
    for i in range(len(data) - 1):
        data[i] = data[i][1:-1:]
    return data


def predict_tomorrow(ticker, units, prediction_days, prediction_day,
                     epoch_par=EPOCHS, batch_size_par=BATCH_SIZE, end_day=test_end):
    try:
        units = int(units)
    except Exception as e:
        print("Units are wrong")
        units = UNITS
    try:
        prediction_days = int(prediction_days)
    except Exception as e:
        print("Prediction Days are wrong")
        prediction_days = PREDICTION_DAYS
    try:
        prediction_day = int(prediction_day)

    except Exception as e:
        print("Prediction Day is wrong")
        prediction_day = PREDICTION_DAY
    data = get_historical_data(ticker, end=end_day)
    all_in_one = get_my_x_axis()
    for i in range(len(data[What])):
        x = 0
        for j in all_in_one.keys():
            if j in POSSIBLE.keys():
                all_in_one[j].append(data[j][i])
                x += (data[j][i])
        all_in_one['everything'].append(x)

    data = pd.DataFrame(data=all_in_one)
    print(data)
    # data.append("all", ignore_index=True)
    # print(data["all"])

    # print(123, data['everything'].values.reshape(-1, 1))
    scalar = MinMaxScaler(feature_range=(0, 1))
    scaled_data_open = scalar.fit_transform(data['open'].values.reshape(-1, 1))
    scaled_data_close = scalar.fit_transform(data['close'].values.reshape(-1, 1))
    scaled_data_low = scalar.fit_transform(data['low'].values.reshape(-1, 1))
    scaled_data_high = scalar.fit_transform(data['high'].values.reshape(-1, 1))
    scaled_data_volume = scalar.fit_transform(data['volume'].values.reshape(-1, 1))
    scaled_data_adjclose = scalar.fit_transform(data['adjclose'].values.reshape(-1, 1))
    scaled_data = []
    for i in range(len(scaled_data_close)):
        scaled_data.append([scaled_data_open[i][-1], scaled_data_close[i][-1],
                            scaled_data_high[i][-1],  scaled_data_low[i][-1], scaled_data_adjclose[i][-1],scaled_data_volume[i][-1]
                               ])
    # scaled_data = scalar.fit_transform(data['everything'].values.reshape(-1, 1))
    scaled_data = scalar.fit_transform(scaled_data)
    scaled_data_y = scalar.fit_transform(data[What].values.reshape(-1, 1))
    x_train = []
    y_train = []
    # print(scaled_data)
    for x in range(prediction_days, len(scaled_data)):
        # print('new', scaled_data[x - prediction_days:x, 0])
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data_y[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units))
    model.add(Dropout(0.2))
    model.add(Dense(units=prediction_day))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, epochs=epoch_par, batch_size=batch_size_par)

    """"" Test The Data """""

    test_data = get_historical_data(ticker, end=end_day)

    all_in_one2 = get_my_x_axis()
    for i in range(len(test_data[What])):
        x = 0
        for j in all_in_one.keys():

            if j != 'everything':
                all_in_one2[j].append(data[j][i])
                x += test_data[j][i]
        all_in_one2['everything'].append(x)
    print(all_in_one2)
    test_data = pd.DataFrame(data=all_in_one2)

    actual_prices = test_data[What].values

    total_data_set = pd.concat((data['open'], test_data[What]), axis=0)
    model_input_open = total_data_set[len(total_data_set) - len(test_data) - prediction_days:].values
    model_input_open = model_input_open.reshape(-1, 1)

    total_data_set = pd.concat((data['close'], test_data[What]), axis=0)
    model_input_close = total_data_set[len(total_data_set) - len(test_data) - prediction_days:].values
    model_input_close = model_input_close.reshape(-1, 1)

    total_data_set = pd.concat((data['low'], test_data[What]), axis=0)
    model_input_low = total_data_set[len(total_data_set) - len(test_data) - prediction_days:].values
    model_input_low = model_input_low.reshape(-1, 1)

    total_data_set = pd.concat((data['high'], test_data[What]), axis=0)
    model_input_high = total_data_set[len(total_data_set) - len(test_data) - prediction_days:].values
    model_input_high = model_input_high.reshape(-1, 1)

    total_data_set = pd.concat((data['adjclose'], test_data[What]), axis=0)
    model_input_adjclose = total_data_set[len(total_data_set) - len(test_data) - prediction_days:].values
    model_input_adjclose = model_input_adjclose.reshape(-1, 1)

    total_data_set = pd.concat((data['volume'], test_data[What]), axis=0)
    model_input_volume = total_data_set[len(total_data_set) - len(test_data) - prediction_days:].values
    model_input_volume = model_input_volume.reshape(-1, 1)

    model_input = []
    for i in range(len(model_input_high)):
        model_input.append([model_input_open[i][-1], model_input_close[i][-1],
                            model_input_high[i][-1], model_input_low[i][-1],
                            model_input_adjclose[i][-1], model_input_volume[i][-1]])

    # model_input = scalar.fit_transform(model_input)
    model_input = scalar.transform(model_input)

    print(model_input[-1], scaled_data[-1])
    # Make Predictions on Test Data

    x_test = []
    for x in range(prediction_days, len(model_input)):
        x_test.append(model_input[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    pre_prices = model.predict(x_test)
    pre_prices = scalar.inverse_transform(pre_prices)

    # Predict Next Day
    print(model_input[-1], model_input[-2], model_input[-3], model_input[-4], model_input[-5],
          model_input[-6], model_input[-7], model_input[-8], model_input[-9], model_input[-10])
    real_data = [model_input[len(model_input) + 1 - prediction_days:len(model_input + 1), 0]]

    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    # print(real_data)
    prediction = model.predict(real_data)
    prediction = scalar.inverse_transform(prediction)
    # # Plot the Test Prediction

    # plt.plot(actual_prices, color='blue')
    # plt.plot(pre_prices, color='red')
    # plt.title(f'{ticker} Share Price')
    # plt.xlabel('Time')
    # plt.ylabel(f'{ticker} Share Price')
    # plt.legend()
    # plt.show()

    return prediction, data[What].values[-1]


def predict_stocks(ticker_list, units='140', prediction_day='1', prediction_days='80',
                   epochs_par=EPOCHS, batch_size=BATCH_SIZE, end_day=test_end):
    long_stocks = []
    short_stocks = []
    float_price = 0
    yesterday = 0
    print("units: ", units, "PD:", prediction_day, "PDS:", prediction_days)
    for my_ticker in ticker_list:
        my_prediction, yesterday = predict_tomorrow(my_ticker, units, prediction_days=prediction_days,
                                                    prediction_day=prediction_day,
                                                    batch_size_par=batch_size, epoch_par=epochs_par, end_day=end_day)
        print(f"Prediction -  {my_prediction, yesterday}")
        print(123456, my_prediction[-1][-1])
        float_price = my_prediction[-1][-1]
        if float_price > float(yesterday):
            long_stocks.append((my_ticker, f"Predict Price - {float_price}", f"Last Price - {float(yesterday)}"))
        else:
            short_stocks.append((my_ticker, f"Predict Price - {float_price}", f"Last Price - {float(yesterday)}"))
    write_in_file(path='Prediction2.txt', data=float_price)
    return float(float_price)

predict_stocks(['NIO'])
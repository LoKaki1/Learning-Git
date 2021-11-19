import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_datareader.data as pdr
import datetime as dt

from tensorflow.compat import v1
from yahoofinancials import YahooFinancials
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

STUPID_IDEA = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.']
test_start = dt.datetime(2010, 1, 1).strftime('%Y-%m-%d')
test_end = (dt.datetime.now() - dt.timedelta(days=0)).strftime('%Y-%m-%d')
MY_X_AXIS = {'open': [], 'close': [], 'high': [], 'low': [], 'volume': [], 'adjclose': []}
POSSIBLE = {'open': [], 'close': [], 'high': [], 'low': [], 'volume': [], 'adjclose': []}

EPOCHS = 25
BATCH_SIZE = 32
UNITS = 140
PREDICTION_DAYS = 80
PREDICTION_DAY = 1

What = 'close'

yf.pdr_override()


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
    return dict((i, []) for i in MY_X_AXIS)


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


def data_checking(units, prediction_days, prediction_day):
    try:
        units = int(units)
    except ValueError:
        print("Units are wrong")
        units = UNITS
    try:
        prediction_days = int(prediction_days)
    except ValueError:
        print("Prediction Days are wrong")
        prediction_days = PREDICTION_DAYS
    try:
        prediction_day = int(prediction_day)

    except ValueError:
        print("Prediction Day is wrong")
        prediction_day = PREDICTION_DAY
    return units, prediction_days, prediction_day


def predicting(ticker, units, prediction_days,
               prediction_day,
               epochs=EPOCHS,
               batch_size=BATCH_SIZE, end_day=test_end):
    """
    This func predicts a stock value in a given day and it gets the parameters for it to work
    """

    """    Checking If Data Is Good   """
    units, prediction_days, prediction_day = data_checking(units, prediction_days, prediction_day)

    data = get_historical_data(ticker, end=end_day)
    all_in_one = get_my_x_axis()
    print(all_in_one)

    """"   Setting Data   """
    for i in range(len(data[What])):
        for j in all_in_one.keys():
            all_in_one[j].append(data[j][i])
    data = pd.DataFrame(data=all_in_one)
    scalar = MinMaxScaler(feature_range=(0, 1))
    my_x = get_my_x_axis()

    t = []
    for i in my_x.keys():
        t.append(scalar.fit_transform(data[i].values.reshape(-1, 1)))

    scaled_data = []
    for i in range(len(t[0])):
        list_all = []
        for j in range(len(t)):
            list_all.append(t[j][i][-1])
        scaled_data.append(list_all)

    scaled_data = scalar.fit_transform(scaled_data)
    scaled_data_y = scalar.fit_transform(data[What].values.reshape(-1, 1))
    x_train = []
    y_train = []
    print(scaled_data)

    for x in range(prediction_days, len(scaled_data)):
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

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    """"" Test The Data """""
    model_input = scaled_data

    """" Make Prediction On Test Data """
    x_test = []
    for x in range(prediction_days, len(model_input)):
        x_test.append(model_input[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    pre_prices = model.predict(x_test)
    pre_prices = scalar.inverse_transform(pre_prices)

    """"  Predict Next Day  """

    real_data = [model_input[len(model_input) + 1 - prediction_days:len(model_input + 1), 0]]

    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    prediction = model.predict(real_data)
    prediction = scalar.inverse_transform(prediction)
    # # Plot the Test Prediction

    plt.plot(y_train, color='blue')
    plt.plot(pre_prices[0], color='red')
    plt.title(f'{ticker} Share Price')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Share Price')
    plt.legend()
    plt.show()
    v1.reset_default_graph()
    return prediction, data[What].values[-1]


def write_in_file(path, data):
    with open(path, 'a') as file:
        file.write(data)
        file.close()


def predict_stocks(ticker, units=UNITS, prediction_day=PREDICTION_DAY, prediction_days=PREDICTION_DAYS,
                   epochs=EPOCHS, batch_size=BATCH_SIZE, end_day=test_end):
    long_stocks = []
    short_stocks = []
    float_price = 0
    yesterday = 0
    print("units: ", units, "PD:", prediction_day, "PDS:", prediction_days)

    my_prediction, yesterday = predicting(ticker, units, prediction_days=prediction_days,
                                          prediction_day=prediction_day,
                                          batch_size=batch_size, epochs=epochs, end_day=end_day)
    print(f"Prediction -  {my_prediction, yesterday}")
    float_price = my_prediction[-1][-1]
    if float_price > float(yesterday):
        long_stocks.append((ticker, f"Predict Price - {float_price}", f"Last Price - {float(yesterday)}"))
    else:
        short_stocks.append((ticker, f"Predict Price - {float_price}", f"Last Price - {float(yesterday)}"))

    write_in_file(path='Prediction.txt', data=''.join([f"\n {str(float_price)}", ", date ", end_day]))
    return float(float_price)


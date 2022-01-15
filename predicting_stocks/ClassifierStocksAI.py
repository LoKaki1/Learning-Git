"""
Classifier stocks prediction that predict stock in a specific day
"""

import numpy as np
import pandas as pd
import datetime as dt

from yahoofinancials import YahooFinancials
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.backend import clear_session
from Common import return_json_data, write_in_file, plot
from Trading.data_order_something import read_data
import os

""" Load Data """
START_INT = 600
STOP_INT = -1
TICKER = 'NIO'
X_VALUES = ['open', 'low', 'high', 'close', ]
START = dt.datetime(2020, 3, 15).strftime('%Y-%m-%d')
END = (dt.datetime.now() - dt.timedelta(days=0)).strftime('%Y-%m-%d')
END_TEST = (dt.datetime.now() - dt.timedelta(days=2)).strftime('%Y-%m-%d')
PREDICTION_DAYS = 30
UNITS = 50
PREDICTION_DAY = 5
DENSE_UNITS = 0.2
EPOCHS = 12
BATCH_SIZE = 64
PARAMETERS = [EPOCHS, UNITS, PREDICTION_DAYS, PREDICTION_DAY]


def read_csv(path, ticker=None):
    return pd.read_csv(path) if os.path.exists(path) else pd.read_csv(read_data(ticker))


def __get_data(data):
    return [[data[key][index]
             for key in X_VALUES]
            for index, i in enumerate(data['close'])]


TICKER_HISTORICAL_DATA = {}
""" Prepare Data """


def get_historical_data(ticker, start=START, end=END):
    return (pd.DataFrame(
        YahooFinancials(ticker).get_historical_price_data(
            start_date=start if start is not None else START,
            end_date=end if end is not None else END,
            time_interval='daily')[
            ticker]['prices']).drop('date', axis=1).set_index('formatted_date'))


def generate_data(*args, ticker):
    json_data = return_json_data(ticker)
    for index, i in enumerate(json_data):
        json_data[index] = args[index] if args[index] is not None else i
        json_data[index] = i if i is not None else PARAMETERS[index]
    return json_data


def get_data(ticker, start_day, end_day, daily):
    """

    :param daily: pass
    :param ticker: stock to get its historical data
    :param start_day: the date that from that you take historical data
    :param end_day: the date that until it, you take historical data
    (start_day, end_day) require internet connection
    :return: Historical data of a stock and divide it into lists that each contains [open, close, high, low]
    """
    global TICKER_HISTORICAL_DATA
    data = get_historical_data(ticker, start_day, end_day)
    if daily:
        TICKER_HISTORICAL_DATA[ticker] = __get_data(data) \
            if ticker not in TICKER_HISTORICAL_DATA else TICKER_HISTORICAL_DATA[ticker]

    else:
        print('You want intraday')
        TICKER_HISTORICAL_DATA[ticker] = __get_data(read_csv(f'../Trading/Historical_data/{ticker}.csv', ticker))
    return TICKER_HISTORICAL_DATA[ticker]


def fit_data(ticker, start_day, end_day, daily):
    """
        func that sets the data to be between 0 and 1 means (40, 10) = (0.123, 0.01) something like that
        :returns the data after fitting it into numbers between 0 and 1
    """
    train_data = get_data(ticker, start_day, end_day, daily)
    """ Making data without lists because scaled data cant
     use lists so data before = [[1, 2, 3, ...], [2, 3, 4, ...] ...] data after = [1, 2, 3, 2, 3, 4 ...] """

    data = np.array([t for i in train_data
                     for t in i]).reshape(-1, 1)

    "Reshape so it matches with scalar api"
    scalar = MinMaxScaler(feature_range=(0, 1))
    """ Fits x values of data (now it makes the real values ) """
    scaled_data = scalar.fit_transform(data)
    return scaled_data, scalar


def prepare_data(scaled_data, prediction_days, prediction_day):
    """ func to prepare data that in x_train it contains prediction_days values and in y_train the predicted price"""
    x_train = []
    y_train = []
    print(prediction_days, prediction_day)
    delta = len(X_VALUES) * prediction_days
    length_const = len(X_VALUES)
    """ Means to start counting from prediction_days index 'til the end """
    for x in range(delta, len(scaled_data) - ((prediction_day - 1) * length_const), length_const):
        """ x_train[0] = array[scaled_data[0], scaled_data[1], ... scaled_data[prediction_days]]
            x_train[1] = array[scaled_data[1], scaled_data[2], ... scaled_data[prediction_days + 1]]
            ...
            x_train[n] = array[scaled_data[n], scaled_data[n + 1] ... scaled_data[prediction_days + n] 

            we make the x_train data that for each y value there
             prediction days values that it can base its prediction on
        """

        x_train.append(scaled_data[x - delta: x, 0])
        """ Remember I changed to discover open to match test model + 0 = open + 1 = low + 2 = high + 3 = close"""
        y_train.append(scaled_data[x + 3: x + (prediction_day * length_const) + 3: length_const, 0][-1])
    """ Reshape the arrays that
    x_train.shape[0] = length of big array 

    x_train[n] = [x_train[n][0], x_train[n][1], ... x_train[n][prediction_days]]"""
    check_data(x_train, y_train, prediction_day)
    x_train, y_train = np.array(x_train), np.array(y_train).reshape(-1, 1)
    """  x_train.shape[0] = the length of the array, x_train.shape[1] =  prediction days 
     means to create a shape with length of x_train.len and width of prediction days on one dimension
    """
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train


def check_data(x_train, y_train, constant=1):
    for i in range(0, len(x_train) - constant):
        if y_train[i] != x_train[i + constant][-1]:
            raise InterruptedError("something went wrong in the code please check it")


def build_model(x_train,
                y_train,
                units,
                epochs,
                ticker=None,
                saved_model=True,
                use_saved_model=False
                ):
    """ Build Model """
    """ Clear session """
    clear_session()

    """ Building The Model """
    if use_saved_model:
        return __load_model(ticker)

    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(DENSE_UNITS))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(DENSE_UNITS))
    model.add(LSTM(units=units))
    model.add(Dropout(DENSE_UNITS))

    """ Returns only one value """
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    """ Fitting x_train to y_train, that makes
     a function that has x values and y values 
      example:  
            x_train =  (23, 24, 25, 26, 123123 ... * (prediction_days)) * all_data
            y_train = (1) ...* all_data - create a func that x[n] = y[n]    """
    model.summary()
    model.fit(x_train, y_train,
              epochs=epochs, batch_size=BATCH_SIZE, verbose='auto', )

    if saved_model:
        model.save(f'saved_model/{ticker}_model')

    return model


def return_test_data(test_start, test_end, prediction_days, prediction_day, ticker, scalar, daily):
    test_data_time = (dt.datetime.strptime(test_start, '%Y-%m-%d') -
                      dt.timedelta(days=prediction_days)).strftime('%Y-%m-%d')

    test_data = pd.DataFrame(get_data(ticker, start_day=test_data_time, end_day=test_end, daily=daily), ).values
    actual_data = []
    model_inputs = test_data.reshape(-1, 1)
    x_test = []
    length = len(X_VALUES)
    delta = length * prediction_days

    model_inputs = scalar.transform(model_inputs)

    for i in range(delta, len(model_inputs) - ((prediction_day - 1) * length), length):
        x_test.append(model_inputs[i - delta: i, 0])
        actual_data.append(model_inputs[i - 6: i - 6 + (prediction_day * length): length, 0][0])
    return x_test, actual_data


def test_model_func(ticker,
                    scalar,
                    model,
                    prediction_days,
                    prediction_day,
                    daily,
                    test_start=dt.datetime(2020, 10, 1).strftime('%Y-%m-%d'),
                    test_end=END_TEST,
                    ):
    """ Test Model
    This part is seeing how accuracy the model on a data that exists but wasn't on it's training"""
    x_test, actual_data = return_test_data(test_start, test_end, prediction_days, prediction_day, ticker, scalar, daily)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    actual_data = np.array(actual_data).reshape(-1, 1)
    actual_data = scalar.inverse_transform(actual_data)
    predicted_prices = model.predict(x_test)
    predicted_prices = scalar.inverse_transform(predicted_prices)
    """ Check len of data is same in both of them """
    print(len(predicted_prices), len(predicted_prices[-1]), len(actual_data))
    pt = []
    for i in predicted_prices:
        pt.append(i[-1])

    pt = np.array(pt)
    return pt, actual_data


def plot_two_graphs(predicted_prices, real_prices, ticker):
    """ func to graph the predicted prices vs the real prices of a stock,
        that way you can see visually how accuracy the model is :)
    """
    predicted_prices = np.array(predicted_prices) if predicted_prices is list else predicted_prices
    real_prices = np.array(real_prices) if real_prices is list else real_prices
    plot(predicted_prices, real_prices, ticker)


def accuracy_ratio(predicted_price, actual_data):
    return sum([min(t / actual_data[i],
                    actual_data[i] / t)
                for i, t in enumerate(predicted_price)]) / len(predicted_price)


def predict_data(scaled_data, scalar, model, prediction_days, prediction_day):
    """ Setting model inputs to be equal to scaled data...
        reason for that is because I want to use the same training data to
        prediction data which makes the neural network gets smarter every day, because it uses new data
    """
    model_inputs = scaled_data
    """
    real_data = last prediction_days values of scaled data 
    """
    real_data = [model_inputs[len(model_inputs) -
                              prediction_days * len(X_VALUES): len(model_inputs) + prediction_day, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    """ After we took the last prediction_days values, we give this x value 
    to predict function and returns the next value according
     to their value and the func it creates in the fit method """
    prediction = model.predict(real_data)

    prediction = scalar.inverse_transform(prediction)
    return prediction


def predict_stock_price_at_specific_day(ticker,
                                        prediction_day=None,
                                        prediction_days=None,
                                        units=None,
                                        epochs=None,
                                        start_day=START,
                                        end_day=END,
                                        model_and_its_args=None,
                                        daily=True):
    """ return predicted stock price in a specific day

        param
            ticker: name of the stock you predict
            prediction_day: what day from you predict
            prediction_days: how many days each prediction is based on
            units: how many units it based on
            dense_units: kill me now
            epochs: how many times it moves on the data
            start_day: from what day you want the data to be based on
                each day y i


    """

    model, \
        scalar, \
        scaled_data, \
        epochs, units, \
        prediction_days, \
        prediction_day, daily = preparation_for_machine(ticker,
                                                        epochs,
                                                        prediction_days, units,
                                                        prediction_day, start_day,
                                                        end_day, model_and_its_args, daily)

    price = predict_data(scaled_data, model=model,
                         prediction_days=prediction_days,
                         scalar=scalar, prediction_day=prediction_day)
    if daily:
        end_day_predicted = (dt.datetime.strptime(end_day, '%Y-%m-%d') +
                             dt.timedelta(days=prediction_day)).strftime('%Y-%m-%d')
    else:
        end_day_predicted = dt.datetime.now() + dt.timedelta(minutes=prediction_day)
    write_in_file('prediction.txt', ''.join(['\n', str(price[-1][-1]), ' ', str(end_day_predicted)]))
    return price


def preparation_for_machine(ticker,
                            epochs,
                            prediction_days, units,
                            prediction_day, start_day,
                            end_day, model_and_its_args, daily,):
    if model_and_its_args is not None:
        return model_and_its_args
    epochs, units, prediction_days, prediction_day, \
        scalar, scaled_data, \
        x_train, y_train = generate_fit_and_prepare_data(ticker,
                                                         epochs,
                                                         prediction_days,
                                                         units, prediction_day,
                                                         start_day, end_day, daily)
    model = build_model(x_train, y_train, units=units,
                        epochs=epochs, ticker=ticker, )
    return model, scalar, scaled_data, epochs, units, prediction_days, prediction_day, daily


def test_model(ticker,
               prediction_day=None,
               prediction_days=None,
               units=None,
               epochs=None,
               start_day=None,
               end_day=None, model_and_its_args=None, daily=True, ):
    """
    function to test the model by making
    prediction on existing data that wasn't given for the model,
    it returns the real values of those days and the predicted values
    each prediction is predicted by the model and giving the model the x value which is the
    prediction_days before that day

    For example -
        prices = [1, 2, ... prediction_days]
        predicted_price = model.predict(prices)
    :returns -
        predicted_prices = [[first] [second]...[last]]
        actual_prices = [[first]
                         [second]
                          ...
                          [last]]
    """

    model, \
        scalar, \
        scaled_data, \
        epochs, units, \
        prediction_days, \
        prediction_day, daily = preparation_for_machine(ticker,
                                                        epochs,
                                                        prediction_days, units,
                                                        prediction_day, start_day,
                                                        end_day, model_and_its_args, daily, )
    return test_model_func(model=model, scalar=scalar, ticker=ticker, prediction_day=prediction_day,
                           prediction_days=prediction_days, daily=daily)


def dumb_test_model(ticker='NIO'):
    return test_model(ticker, units=1, prediction_days=21, epochs=1)


def generate_fit_and_prepare_data(ticker, epochs, prediction_days, units, prediction_day, start_day, end_day, daily):
    epochs, units, prediction_days, prediction_day = generate_data(epochs,
                                                                   prediction_days,
                                                                   units,
                                                                   prediction_day,
                                                                   ticker=ticker,
                                                                   )
    scaled_data, scalar = fit_data(ticker, start_day=start_day, end_day=end_day, daily=daily)
    x_train, y_train = prepare_data(scaled_data, prediction_days, prediction_day)
    return epochs, units, prediction_days, prediction_day, scalar, scaled_data, x_train, y_train


def build_model_for_multiple_prediction(ticker, prediction_day=None,
                                        prediction_days=None,
                                        units=None,
                                        epochs=None,
                                        start_day=START,
                                        end_day=END,
                                        daily=True):
    """
    function to create model for multiple predictions the model here
     is automatically false because we can't create model from model
    """

    return preparation_for_machine(ticker,
                                   epochs,
                                   prediction_days, units,
                                   prediction_day, start_day,
                                   end_day, None, daily=daily)


def __load_model(ticker, ):
    """
    Function to load model from saved model file
    """

    return load_model(path) if os.path.exists((path := f'saved_model/{ticker}_model')) else None


def main():
    ticker = 'RELI'
    model_and_its_args = build_model_for_multiple_prediction(ticker, daily=False, )


if __name__ == '__main__':
    main()

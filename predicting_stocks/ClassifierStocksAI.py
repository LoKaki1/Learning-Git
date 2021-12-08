"""
Classifier stocks prediction that predict stock in a specific day
"""

import numpy as np
import pandas as pd
import datetime as dt
import ast


from yahoofinancials import YahooFinancials
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from Common import write_in_file, plot

""" Load Data """
START_INT = 600
STOP_INT = -1
TICKER = 'NIO'
X_VALUES = ['open', 'close', 'low', 'high', ]
START = dt.datetime(2019, 12, 1).strftime('%Y-%m-%d')
END = (dt.datetime.now() - dt.timedelta(days=0)).strftime('%Y-%m-%d')
PREDICTION_DAYS = 21
UNITS = 51
PREDICTION_DAY = 1
DENSE_UNITS = 0.2
EPOCHS = 25
BATCH_SIZE = 64

""" Prepare Data """


def get_historical_data(ticker, start=START, end=END):
    ticker = ticker.strip("'")
    data = YahooFinancials(ticker)
    data = data.get_historical_price_data(start, end, 'daily')
    if 'prices' in pd.DataFrame(data[ticker]).keys():
        t_data = pd.DataFrame(data[ticker]['prices'])
        t_data = t_data.drop('date', axis=1).set_index('formatted_date')
        t_data.head()
    else:
        data = data.get_historical_price_data(dt.datetime(2020, 3, 1).strftime('%Y-%m-%d'), end, 'daily')
        t_data = pd.DataFrame(data[ticker]['prices'])
        t_data = t_data.drop('date', axis=1).set_index('formatted_date')
        t_data.head()
    return t_data


def get_data(ticker, start_day=START, end_day=END, int_start_day=0, int_end_day=0):
    """

    :param int_end_day: Where to stop give historical data without internet connection
    :param int_start_day: Where to start give the historical data without connection to internet
    :param ticker: stock to get its historical data
    :param start_day: the date that from that you take historical data
    :param end_day: the date that until it you take historical data
    (start_day, end_day) require internet connection
    :return: Historical data of a stock and divide it into lists that each contains [open, close, high, low]
    """
    print(ticker)
    try:
        data = get_historical_data(ticker, start_day, end_day)
    except OSError:
        print('Can\'t connect to internet, use saved data instead')

        data = get_data_from_saved_file(ticker)[ticker]
        return [[data[key][i]
                 for key in X_VALUES] for i in range(len(data['close']))][int_start_day:int_end_day]
    return [[data[key][i]
             for key in X_VALUES]
            for i in range(len(data['close']))]


def fit_data(ticker, train_data=None, start_day=START, end_day=END, start_int=START_INT, stop_int=STOP_INT):
    """ func that sets the data to be between 0 and 1 means (40, 10) = (0.123, 0.01) something like that
        :returns the data after fitting it into numbers between 0 and 1
    """
    train_data = get_data(ticker, start_day, end_day, start_int, stop_int) if train_data is None else train_data
    """ Making data without lists because scaled data cant
     use lists so data before = [[1, 2, 3, ...], [2, 3, 4, ...] ...] data after = [1, 2, 3, 2, 3, 4 ...] """

    data = []
    for i in train_data:
        for t in i:
            data.append(t)
    "Reshape so it matches with scalar api"
    data = np.array(data).reshape(-1, 1)
    scalar = MinMaxScaler(feature_range=(0, 1))
    """ Fits x values of data (now it makes the real values ) """
    print(data)
    scaled_data = scalar.fit_transform(data)
    return scaled_data, scalar


def prepare_data(scaled_data, prediction_days=PREDICTION_DAYS, ):
    """ func to prepare data that in x_train it contains prediction_days values and in y_train the predicted price"""
    x_train = []
    y_train = []
    delta = len(X_VALUES) * prediction_days
    length_const = len(X_VALUES)
    """ Means to start counting from prediction_days index 'til the end """
    for x in range(delta, len(scaled_data) - 1, length_const):
        """ x_train[0] = array[scaled_data[0], scaled_data[1], ... scaled_data[prediction_days]]
            x_train[1] = array[scaled_data[1], scaled_data[2], ... scaled_data[prediction_days + 1]]
            ...
            x_train[n] = array[scaled_data[n], scaled_data[n + 1] ... scaled_data[prediction_days + n] 

            we make the x_train data that for each y value there
             prediction days values that it can base its prediction on
        """
        x_train.append(scaled_data[x - delta: x, 0])
        y_train.append(scaled_data[x + 1, 0])

    """ Reshape the arrays that
    x_train.shape[0] = length of big array 

    x_train[n] = [x_train[n][0], x_train[n][1], ... x_train[n][prediction_days]]"""

    x_train, y_train = np.array(x_train), np.array(y_train)
    """  x_train.shape[0] = the length of the array, x_train.shape[1] =  prediction days 
     means to create a shape with length of x_train.len and width of prediction days on one dimension
    """
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    return x_train, y_train


def build_model(x_train, y_train,
                units=UNITS,
                prediction_day=PREDICTION_DAY,
                dense_units=DENSE_UNITS, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """ Build Model """

    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(dense_units))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dense_units))
    model.add(LSTM(units=units))
    model.add(Dropout(dense_units))
    model.add(Dense(units=prediction_day))

    model.compile(optimizer='adam', loss='mean_squared_error')
    """ Fitting x_train to y_train, that makes
     a function that has x values and y values 
      example:  
            x_train =  (23, 24, 25, 26, 123123 ... * (prediction_days)) * all_data
            y_train = (1) ...* all_data - create a func that x[n] = y[n]    """
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model


def test_model_func(ticker, scalar, model, prediction_days,
                    test_start=dt.datetime(2019, 8, 1).strftime('%Y-%m-%d'),
                    test_end=dt.datetime(2020, 5, 1).strftime('%Y-%m-%d')):
    """ Test Model
    This part is seeing how accuracy the model on a data that exists but wasn't on it's training"""
    int_start = START_INT
    int_stop = STOP_INT
    try:
        t_data = get_historical_data(ticker, test_start, test_end)
        actual_data = t_data['close'].values
    except OSError:
        print('still cannot connect to internet using saved data instead')
        t_data = get_data_from_saved_file(ticker, )[ticker]
        actual_data = t_data['close'][int_start: int_stop]
    """
    Combining test data dates prices with the 
    start - prediction_days,
     to get the x values of the prediction from the start
    """
    test_data = get_data(ticker, start_day=test_start, end_day=test_end, int_start_day=int_start, int_end_day=int_stop)
    test_data_time = (dt.datetime.strptime(test_start, '%Y-%m-%d') -
                      dt.timedelta(days=prediction_days)).strftime('%Y-%m-%d')
    data = get_data(ticker,
                    start_day=test_data_time, end_day=test_start, int_start_day=int_start - prediction_days,
                    int_end_day=int_start)

    """  what we input to model to get output after the training """
    """ This makes that model model_inputs[n + prediction_days] = test_close_data[n] """
    total_data_test = pd.concat((pd.DataFrame(data), pd.DataFrame(test_data)), axis=0)
    model_inputs = total_data_test.values
    """ Reminder reshape means what shape it looks like.. it doesn't change the order """
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scalar.transform(model_inputs)

    """ Make Prediction On Test Data (Doesn't effect on the real prediction) """
    x_test = []
    for x in range(prediction_days * len(X_VALUES), len(model_inputs) - 1, len(X_VALUES)):
        x_test.append(model_inputs[x - prediction_days * len(X_VALUES): x, 0])

    """ Reshape x_test so will look [1, 2, 3, 4, 5..prediction_days],...length times
    
    """
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    """ Predict On Existing data """

    predicted_price = model.predict(x_test)
    predicted_price = predicted_price.reshape(-1, 1)

    predicted_price = scalar.inverse_transform(predicted_price)

    print(len(predicted_price), len(actual_data))

    return predicted_price, actual_data


def plot_two_graphs(predicted_prices, real_prices, ticker):
    """ func to graph the predicted prices vs the real prices of a stock,
        that way you can see visually how accuracy the model is :)
    """
    predicted_prices = np.array(predicted_prices) if predicted_prices is list else predicted_prices
    real_prices = np.array(real_prices) if real_prices is list else real_prices
    plot(predicted_prices, real_prices, ticker)


def accuracy_ratio(predicted_price, actual_data):
    return sum([min(predicted_price[i] / actual_data[i],
                    actual_data[i] / predicted_price[i])
                for i, t in enumerate(predicted_price)]) / len(predicted_price)


def predict_data(scaled_data, scalar, model, prediction_day=PREDICTION_DAY, prediction_days=PREDICTION_DAYS):
    """ Setting model inputs to be equal to scaled data...
        reason for that is because I wanna use the same training data to
        prediction data which makes the neural network gets smarter everyday, because it uses new data
    """
    model_inputs = scaled_data
    """
    real_data = last prediction_days values of scaled data 
    """
    real_data = [model_inputs[len(model_inputs) + prediction_day -
                              prediction_days: len(model_inputs + prediction_day), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
    """ After we took the last prediction_days values, we give this x value 
    to predict function and returns the next value according
     to their value and the func it creates in the fit method """
    prediction = model.predict(real_data)

    prediction = scalar.inverse_transform(prediction)
    return prediction


def predict_stock_price_at_specific_day(ticker,
                                        prediction_day=PREDICTION_DAY,
                                        prediction_days=PREDICTION_DAYS,
                                        units=UNITS,
                                        dense_units=DENSE_UNITS,
                                        epochs=EPOCHS,
                                        start_day=START,
                                        end_day=END,
                                        batch_size=BATCH_SIZE,
                                        model=None):
    """ :return predicted stock price in a specific day

        :param
            ticker: name of the stock you predict
            prediction_day: what day from you predict
            prediction_days: how many days each prediction is based on
            units: how many units it based on
            dense_units: kill me now
            epochs: how many times it moves on the data
            start_day: from what day you want the data to be based on
                each day y i


     """

    scaled_data, scalar = fit_data(ticker, start_day=start_day, end_day=end_day)
    x_train, y_train = prepare_data(scaled_data, prediction_days)
    model = build_model(x_train, y_train, units=units, prediction_day=prediction_day, dense_units=dense_units,
                        epochs=epochs, batch_size=batch_size) if model is None else model

    price = predict_data(scaled_data, model=model,
                         prediction_day=prediction_day,
                         prediction_days=prediction_days,
                         scalar=scalar)
    write_in_file('prediction.txt', ''.join(['\n', str(price[-1][-1]), str(end_day)]))
    return price


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


def test_model(ticker, prediction_day=PREDICTION_DAY,
               prediction_days=PREDICTION_DAYS,
               units=UNITS,
               dense_units=DENSE_UNITS, epochs=EPOCHS,
               end_day=END, batch_size=BATCH_SIZE, model=None):
    """
    function to test the model by making
    prediction on existing data that wasn't given for the model,
    it returns the real values of those days and the predicted values
    each prediction is predicted by the model and giving the model the x value which is the
    prediction_days before that day

    For example -
        prices = [1, 2, .. prediction_days]
        predicted_price = model.predict(prices)
    :returns -
        predicted_prices = [[first] [second]..[last]]
        actual_prices = [[first]
                         [second]
                          ..
                          [last]]
    """
    scaled_data, scalar = fit_data(ticker, end_day=end_day)
    x_train, y_train = prepare_data(scaled_data, prediction_days)
    model = build_model(x_train, y_train, units=units, prediction_day=prediction_day, dense_units=dense_units,
                        epochs=epochs, batch_size=batch_size) if model is None else model
    return test_model_func(model=model, scalar=scalar, ticker=ticker, prediction_days=prediction_days, )


def predict_stocks_avg(ticker, prediction_day=PREDICTION_DAY,
                       prediction_days=PREDICTION_DAYS,
                       units=UNITS,
                       dense_units=DENSE_UNITS, avg=4):
    number = sum(
        [predict_stock_price_at_specific_day(ticker, prediction_day=prediction_day, prediction_days=prediction_days,
                                             units=units, dense_units=dense_units
                                             ) in range(avg)]) / avg
    write_in_file(path='Data/avg.txt', data=str(number))
    return number


def dumb_test_mode():
    return test_model('NIO', units=1, prediction_days=2)


def build_model_for_multiple_prediction(ticker, prediction_day=PREDICTION_DAY,
                                        prediction_days=PREDICTION_DAYS,
                                        units=UNITS,
                                        dense_units=DENSE_UNITS, epochs=EPOCHS,
                                        end_day=END, batch_size=BATCH_SIZE, ):
    scaled_data, scalar = fit_data(ticker, end_day=end_day)
    x_train, y_train = prepare_data(scaled_data, prediction_days)
    return build_model(x_train, y_train, units=units, prediction_day=prediction_day, dense_units=dense_units,
                       epochs=epochs, batch_size=batch_size)


def main():
    ticker = 'NIO'
    model = build_model_for_multiple_prediction(ticker, epochs=19, units=48, prediction_days=13, dense_units=0.2)
    # t, y = test_model('RIOT', units=51, prediction_days=21, model=model)
    # plot_two_graphs(t, y, 'RIOT')
    # print(accuracy_ratio(t, y))
    print(predict_stock_price_at_specific_day(ticker, model=model), get_historical_data(ticker)['close'][-1])


if __name__ == '__main__':
    main()

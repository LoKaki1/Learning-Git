import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_datareader.data as pdr
import datetime as dt
import whatsapp
from yahoofinancials import YahooFinancials
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
# from Classfier_AI import get_historical_data

test_start = dt.datetime(2015, 1, 1).strftime('%Y-%m-%d')
test_end = dt.datetime.now().strftime('%Y-%m-%d')
MY_X_AXIS = {'open': [], 'close': [], 'high': [], 'low': [],  'everything': []}
POSSIBLE = {'open': [], 'close': [], 'high': [], 'low': []}

EPOCHS = 25
BATCH_SIZE = 64
units = 130
prediction_days = 35


What = 'close'
prediction_day = 1
yf.pdr_override()


def get_historical_data(ticker, start=test_start, end=test_end):
    ticker = ticker.strip("'")
    data = YahooFinancials(ticker)
    print(ticker)
    data = data.get_historical_price_data(start, end, 'daily')

    t_data = pd.DataFrame(data[ticker]['prices'])
    t_data = t_data.drop('date', axis=1).set_index('formatted_date')
    t_data.head()

    return t_data


def getting_stocks():
    file = open('../Trading/Data/stocks.txt', 'r')
    data = file.read()
    file.close()
    print(data)

    data = data.strip(']').strip('[').split(', ')
    for i in range(len(data) - 1):
        data[i] = data[i][1:-1:]
    return data


def predict_tomorrow(ticker, ):
    data = get_historical_data(ticker)
    all_in_one = {'open': [], 'close': [], 'high': [], 'low': [], 'everything': [],}
    for i in range(len(data[What])):
        x = 0
        for j in all_in_one.keys():

            if j in POSSIBLE:
                all_in_one[j].append(data[j][i])
                x += data[j][i]
        all_in_one['everything'].append(x)
    print(all_in_one)
    data = pd.DataFrame(data=all_in_one)

    # data.append("all", ignore_index=True)
    # print(data["all"])

    # print(x.head())
    scalar = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scalar.fit_transform(data['everything'].values.reshape(-1, 1))
    scaled_data_y = scalar.fit_transform(data[What].values.reshape(-1, 1))
    x_train = []
    y_train = []

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

    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)

    """"" Test The Data """""

    test_data = get_historical_data(ticker)

    all_in_one2 = {'open': [], 'close': [], 'high': [], 'low': [], 'everything': []}
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

    total_data_set = pd.concat((data['everything'], test_data[What]), axis=0)

    model_input = total_data_set[len(total_data_set) - len(test_data) - prediction_days:].values
    model_input = model_input.reshape(-1, 1)
    model_input = scalar.transform(model_input)

    # Make Predictions on Test Data

    x_test = []
    for x in range(prediction_days, len(model_input)):
        x_test.append(model_input[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    pre_prices = model.predict(x_test)
    pre_prices = scalar.inverse_transform(pre_prices)

    # Predict Next Day

    real_data = [model_input[len(model_input) + 1 - prediction_days:len(model_input + 1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

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


def predict_stocks(ticker_list, what=What):
    long_stocks = []
    short_stocks = []

    for my_ticker in ticker_list:

        try:
            my_prediction, yesterday = predict_tomorrow(my_ticker)
            print(f"Prediction -  {my_prediction, yesterday}")

            float_price = float(str(my_prediction).split(' ')[-1][:-3].strip('[').strip(']'))

            if float_price > float(yesterday):
                long_stocks.append((my_ticker, f"Predict Price - {float_price}", f"Last Price - {float(yesterday)}"))
            else:
                short_stocks.append((my_ticker, f"Predict Price - {float_price}", f"Last Price - {float(yesterday)}"))
        except Exception as e:
            print(e.args)

        finally:
            pass

    print("Stocks for Long - ", '\n', "Stocks for Short - ", short_stocks)
    data = whatsapp.write_in_file(long_stocks, short_stocks)
    whatsapp.send_email(body=data)


# def main():
#     ticker_list = getting_stocks()
#     predict_stocks(ticker_list)
#
#
# if __name__ == '__main__':
#     main()


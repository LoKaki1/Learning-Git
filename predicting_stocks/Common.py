import json
import matplotlib.pyplot as plt


def write_in_file(path, data):
    with open(path, 'a') as file:
        file.write(data)
        file.close()


def plot(data, pre_prices, ticker):
    plt.plot(data, color='blue')
    print(pre_prices)
    plt.plot(pre_prices, color='red')
    plt.title(f'{ticker} Share Price')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Share Price')
    plt.legend()
    plt.show()


def write_in_json_file(path, data, ticker=None):
    a_file = open(path, "r")
    json_object = json.load(a_file)
    a_file.close()
    print(json_object)
    json_object[ticker] = data[ticker]

    a_file = open(path, "w")
    json.dump(json_object, a_file)
    a_file.close()
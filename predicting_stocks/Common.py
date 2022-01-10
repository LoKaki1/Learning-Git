import json
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import os
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


def return_json_data(ticker, json_path=r'C:\Users\meir\PycharmProjects\Learning-Git\predicting_stocks\settings_for_ai\parameters_status.json'):
    with open(json_path, 'r'):
        print(json_path)

    if not json_path:
        return None
    with open(json_path, 'r') as json_file:
        p = json.load(json_file)
        if ticker in p:
            p = p[ticker]['settings']
            return [p['epochs'], p['units'], p['prediction_days'], p['prediction_day']]
        else:
            return [None, None, None, None]



def save_model(train_images, train_labels):
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    batch_size = 32

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=5 * batch_size)

return_json_data('NIO')
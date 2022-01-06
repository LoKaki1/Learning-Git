"""
Script to check what's the  best parameters for the classifier
using random values with predicting some prices and compare it with the real values
using test_data() that returns the real values and the predicting values that, Ai
predicted according to its model
"""
from ClassifierStocksAI import test_model, accuracy_ratio, EPOCHS, UNITS, PREDICTION_DAYS
from Common import write_in_file, write_in_json_file
from random import randint, Random

TICKER_LIST = ['LI', 'NIO', 'RIOT', 'AAPL', 'TSLA',]

""" Constant  """
GARBAGE_PATH = r'../predicting_stocks/settings_for_ai/garbage_parameters_status.txt'
PATH = r'../predicting_stocks/settings_for_ai/parameters_status.txt'
JSON_PATH = r'../predicting_stocks/settings_for_ai/parameters_status.json'
BATCH_SIZE = 64
PREDICTION_DAY = 10
DENSE_UNITS = 0.2
LAST_RATIO = 0.95
RATIO = 0.98


def random_double(random_multiplier=1):
    """

    :param random_multiplier: what range you want it for example -
        if random_multiplier = 10 then the range of the float number can be 0.84 or 0.21 etc.
    :return: random float
    """
    return float(f'0.{Random().randint(1, 9 * random_multiplier)}')


def generate_random_values_according_to_ratio(epochs, units, prediction_days):
    """
    first function to generate the numbers according to the last result means 6 options 3!
    x y z
    (x, y, random)
    (x, random, z)
    (random, y, z)
    (random, random, z)
    (random, y, random)
    (x, random, random)
    (x, y, z)
    (random, random, random)

    """
    return [generate_random_values(epochs=epochs, units=units),
            generate_random_values(epochs=epochs, prediction_days=prediction_days),
            generate_random_values(units=units, prediction_days=prediction_days),
            generate_random_values(prediction_days=prediction_days),
            generate_random_values(units=units),
            generate_random_values(epochs=epochs),
            # generate_random_values(epochs, units, prediction_days),
            generate_random_values()
            ]


def generate_random_values(epochs=None, units=None, prediction_days=None, prediction_day=PREDICTION_DAY):
    """ function to generate random values
        [epochs, units, prediction_days, dense_units]
    """
    if not epochs:
        epochs = randint(10, 40)
    if not units:
        units = randint(10, 100)
    if not prediction_days:
        prediction_days = randint(10, 100)

    return epochs, units, prediction_days, prediction_day
    # return 2, 2, 2, DENSE_UNITS


def ratio(ticker, *args):
    """ :return ratio between predicted_ratio to real_values

        :param ticker = the ticker that model is testing
                *args =
                    epochs, units, prediction_days, dense_units
    """
    epochs, units, prediction_days, prediction_day, = args[0]
    print(epochs, units, prediction_days, prediction_day,)
    predicted_values, real_values = test_model(ticker,
                                               epochs=epochs,
                                               units=units,
                                               prediction_days=prediction_days,
                                               prediction_day=PREDICTION_DAY)

    return accuracy_ratio(predicted_values, real_values)


def create_data_object(parameters, ticker, current_ratio):
    return str({
        ticker: {
            "settings":
                {
                    "epochs": parameters[0],
                    "units": parameters[1],
                    "prediction_days": parameters[2],

                    "ratio": current_ratio

                }
        }
    })


def generate_best_values2():
    father_parameters = generate_random_values_according_to_ratio(EPOCHS, UNITS, PREDICTION_DAYS)
    last = None
    best_parameters = None
    for ticker in TICKER_LIST:
        while True:

            children_dict = dict((float(ratio(ticker, child)), child) for index, child in enumerate(father_parameters))
            if last is not None and best_parameters is not None:
                children_dict[last] = best_parameters
            print(children_dict)
            best_child_ratio = max(children_dict.keys())
            best_parameters = children_dict[best_child_ratio]
            
            if best_child_ratio > RATIO:
                child_object = create_data_object(parameters=best_parameters,
                                                  ticker=ticker,
                                                  current_ratio=best_child_ratio)
                write_in_json_file(path=JSON_PATH,
                                   data=child_object,
                                   ticker=ticker)
                print('Not')
                break
            else:
                child_object = '\n' + str(create_data_object(parameters=best_parameters,
                                                             ticker=ticker,
                                                             current_ratio=best_child_ratio))
                write_in_file(path=GARBAGE_PATH, data=child_object, )
                print('I wrote in garbage')
            last = best_child_ratio
            father_parameters = generate_random_values_according_to_ratio(best_parameters[0],
                                                                          best_parameters[1],
                                                                          best_parameters[2])


# rat = 0.98
# par = (25, 100, 79, 0.2)
# ticker = "LI"
# t = create_data_object(par, ticker, rat)
# print(t)
# write_in_json_file('../predicting_stocks/settings_for_ai/parameters_status.json', data=t, ticker=ticker)
generate_best_values2()

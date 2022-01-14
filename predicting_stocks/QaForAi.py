"""
Script to check what's the  best parameters for the classifier
using random values with predicting some prices and compare it with the real values
using test_data() that returns the real values and the predicting values that, Ai
predicted according to its model
"""
from ClassifierStocksAI import test_model, accuracy_ratio
from Common import write_in_file
from random import randint

TICKER_LIST = ['NIO', 'RIOT']

""" Constant  """
GARBAGE_PATH = r'C:\Users\meir\Learning-Git\predicting_stocks\settings_for_ai\garbage_parameters_status.txt'
PATH = r'C:\Users\meir\Learning-Git\predicting_stocks\settings_for_ai\parameters_status.txt'
BATCH_SIZE = 64
PREDICTION_DAY = 15
DENSE_UNITS = 0.2
LAST_RATIO = 0.95


def generate_random_values():
    """ function to generate random values
        [epochs, units, prediction_days, dense_units]
    """
    return randint(10, 30), randint(20, 60), randint(6, 70), DENSE_UNITS


def ratio(ticker, *args):
    """ :return ratio between predicted_ratio to real_values

        :param ticker = the ticker that model is testing
                *args =
                    epochs, units, prediction_days, dense_units
    """
    epochs, units, prediction_days, dense_units,  = args[0]
    predicted_values, real_values = test_model(ticker,
                                               epochs=epochs,
                                               units=units,
                                               prediction_days=prediction_days)
    return accuracy_ratio(predicted_values, real_values)


def generate_best_values():
    """
    function that write the best parameters for AI in PATH file, by running the test
    model function multiple times on different parameters
    """
    last_ratio = LAST_RATIO
    while True:
        """
        This while loop runs all the above function in order to find the closest 
        ratio to the real price data, which that way 
        I can choose which parameters are the most suitable for my ClassifierStocksAi,
        (Which at this time works only on prices without volume)
        """
        parameters = generate_random_values()
        """
        Generate parameters and save it so I can access them later in order to save them
        """

        ticker = TICKER_LIST[randint(0, len(TICKER_LIST) - 1)]
        """
        Pick a random stock because I do not want the parameters to be specific on one stock
        """
        current_ratio = ratio(ticker, parameters)

        " Save parameters if ratio is bigger than the last ratio which means those parameters are better than previous"
        data = "".join([
            "\nticker=", ticker,
            ", epochs=", str(parameters[0]),
            ", units=", str(parameters[1]),
            ", prediction_days=", str(parameters[2]),
            ", dense_units=", str(parameters[3]),
            ",     ratio=", str(current_ratio)
        ])
        """
        Write data at the end if the current ratio is bigger then the last biggest ratio
        """
        if current_ratio > last_ratio:
            last_ratio = current_ratio
            write_in_file(PATH, data)

        else:
            """
            This else is for seeing what data it tried but failed, so I see what isn't working or it tried :)
            """
            write_in_file(GARBAGE_PATH, data)
        """
        It's written at the end because I want the file to updated and then break from the loop
        """
        if current_ratio >= 0.985:
            break


""" Should be fixed, I think it works pair every stock individually
    Solution, Create json file that contains the parameters that suits best for a specific stock 
    
 """


generate_best_values()

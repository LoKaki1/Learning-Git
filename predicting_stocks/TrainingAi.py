from ClassifierStocksAiClass import ClassifierAi
import Common as Cm
import random

"Prediction Day stays at 1"
PARAMS = [(3, 30), (10, 120), (2, 100), ]


def generate_random_numbers(*param):
    return [i if i is not None else random.randint((par := PARAMS[index])[0], par[1]) for index, i in enumerate(param)]


def generate_children(*params):
    return [generate_random_numbers(params[0], None, None),
            generate_random_numbers(None, params[1], None),
            generate_random_numbers(None, None, params[2]),
            generate_random_numbers(params[0], params[1], None),
            generate_random_numbers(params[0], None, params[2]),
            generate_random_numbers(None, params[1], params[2]),
            generate_random_numbers(None, None, None),
            ]


def ratio(params, ticker):
    print(params)
    child = ClassifierAi(ticker=ticker, epochs=params[0], units=params[1], prediction_days=params[2],
                         load_model_from_local=False, daily=True)
    return child.test_model_and_return_accuracy_ratio(),


def create_json_object(ticker, best_child_par):
    return {
        ticker: {
            "settings": {
                        'epochs': best_child_par[0],
                        'units': best_child_par[1],
                        'prediction_days': best_child_par[2]
            }
        }
    }


def main():
    ticker = 'NIO'
    e, u, p = 25, 40, 20
    params = generate_children(e, u, p, )
    father = ClassifierAi(ticker=ticker, epochs=e, units=u, prediction_days=p, load_model_from_local=True, daily=True)
    father.test_model()
    print(params)
    while True:
        children_dict = dict((ratio(i, ticker), i) for i in params)
        children_dict[father.test_model_and_return_accuracy_ratio()] = [father.epochs,
                                                                        father.units,
                                                                        father.prediction_days]
        best_child_par = children_dict[(best_ratio := max(children_dict.keys()))]
        print(best_child_par, best_ratio)
        father = ClassifierAi(ticker=ticker, epochs=best_child_par[0],
                              units=best_child_par[1],
                              prediction_days=best_child_par[2],
                              load_model_from_local=True, daily=True)
        params = generate_children(best_child_par)
        json_object = create_json_object(ticker, best_child_par)
        Cm.write_in_json_file('../predicting_stocks/settings_for_ai/parameters_status.json', data=json_object,
                              ticker=ticker) if best_ratio > 0.985 else Cm.write_in_file(
            '../predicting_stocks/settings_for_ai'
            '/garbage_parameters_status.txt',
            data=json_object)


if __name__ == '__main__':

    main()

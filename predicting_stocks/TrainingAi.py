from ClassifierStocksAiClass import ClassifierAi
import Common as Cm
import random

"Prediction Day stays at 1"
PARAMS = [(3, 30), (10, 120), (2, 100), ]


def generate_random_numbers(*param):
    return [i if i is not None else random.randint((par := PARAMS[index])[0], par[1]) for index, i in enumerate(param)]


def generate_children(epochs, units, prediction_days,):
    return [generate_random_numbers(epochs, None, None),
            generate_random_numbers(None, units,  None),
            generate_random_numbers(None, None, prediction_days,),
            generate_random_numbers(epochs, units, None),
            generate_random_numbers(epochs, None, prediction_days,),
            generate_random_numbers(None, units, prediction_days),
            generate_random_numbers(None, None, None),
            ]


def ratio(epochs, units, prediction_days, ticker):
    print(epochs, units, prediction_days, type(epochs), type(units), type(prediction_days),)
    child = ClassifierAi(ticker=ticker, epochs=epochs, units=units, prediction_days=prediction_days,
                         load_model_from_local=False, daily=True)
    try:
        price = float(child.test_model_and_return_accuracy_ratio()[-1])
    except ValueError:
        price = float(child.test_model_and_return_accuracy_ratio())
    return price


def create_json_object(ticker, best_child_par, __ratio__):
    return {
        ticker: {
            "settings": {
                'epochs': best_child_par[0],
                'units': best_child_par[1],
                'prediction_days': best_child_par[2],
                'ratio': __ratio__

            }
        }
    }


def main():
    tickers = ['BSFC', 'LLNW', 'BON']
    for ticker in tickers:
        e, u, p = 12, 50, 21
        params = generate_children(e, u, p, )
        father = ClassifierAi(ticker=ticker, epochs=e, units=u, prediction_days=p,
                              load_model_from_local=False, daily=False, source='yahoo')
        counter = 0
        while True:
            counter += 1
            children_dict = dict((ratio(i[0], i[1], i[2], ticker), i) for i in params)
            children_dict[(father_ratio := float(father.test_model_and_return_accuracy_ratio()[-1]))] = \
                [father.epochs,
                 father.units,
                 father.prediction_days]
            best_child_par = children_dict[(best_ratio := (max(children_dict.keys())))]
            print(best_child_par, best_ratio)
            father = ClassifierAi(ticker=ticker, epochs=best_child_par[0],
                                  units=best_child_par[1],
                                  prediction_days=best_child_par[2],
                                  load_model_from_local=False,
                                  daily=False, child=str(counter),
                                  source='yahoo')\
                if best_child_par != children_dict[father_ratio] else father

            params = generate_children(best_child_par[0], best_child_par[1], best_child_par[2], )
            json_object = create_json_object(ticker, best_child_par, best_ratio)
            Cm.write_in_json_file('../predicting_stocks/settings_for_ai/parameters_status.json', data=json_object,
                                  ticker=ticker) if float(best_ratio) > 0.985 else Cm.write_in_file(
                '../predicting_stocks/settings_for_ai'
                '/garbage_parameters_status.txt',
                data=''.join(['\n', str(json_object)]))
            if float(best_ratio) > 0.985:
                break


if __name__ == '__main__':
    main()

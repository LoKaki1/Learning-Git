from random import randint, getrandbits
import itertools
from ClassifierStocksAiClass import ClassifierAi
import Common as Cm

"""
['epochs', 'units', 'prediction_days', {
'layers': {
    1: ['Dense', '**kwargs'], 2: ['LSTM', '**kwargs'], 3: ['Dropout', '**kwargs']},
    'activation': [
        'deserialize', 'elu', 'exponential', 'gelu', 'get',
        'hard_sigmoid', 'linear', 'relu', 'selu', 'serialize', 'sigmoid',
        'softmax', 'softplus', 'softsign', 'swish', 'tanh'
        ]
            }
compile: {
        optimizer: [
            sgd, rmsprop, adam, adadelta, adagrad, adamax, nadam, ftrl
        ],
        loss: [
                mean_squared_error,
                mean_absolute_error,
                mean_absolute_percentage_error,
                mean_squared_logarithmic_error,
                cosine_similarity,
                huber,
                log_cosh,

        ],
        metrics: [
                    AUC,
                    Precision,
                    Recall,
                    TruePositives,
                    TrueNegatives,
                    FalsePositives,
                    FalseNegatives,
                    PrecisionAtRecall,
                    SensitivityAtSpecificity,
                    SpecificityAtSensitivity,

        ]
}
"""

"""
main (SETTINGS: list = None)
    [settings1, settings2, ... settings(len(settings))] = generator(SETTINGS) - 
        (function that generate all random settings
            with every combination can be - means first the settings looks like this (1, 2, 3)
            it return every combination that can be by leaving some numbers to stay the
            same in every child of those settings.
            EXAMPLE -> (1, 2, 3)
                        [(1, random, random), (1, 2, random),
                         (1, random, 3), (random, 2, random),
                         (random, random, 3), (random, 2, 3)
                        ]
        After generate settings creates ClassifierStockAI with all those settings,
        (create a way to deliver list of layers into build model)
        use test_model function in order to get the best out of the model,
        create a dictionary that contains each of the settings + its accuracy ratio {1: 0.234, ...}
        set this child to be the next father and start looping using same function
        )
"""

LAYERS = ('Dense', 'LSTM', 'Dropout')
ACTIVATION = [
    'relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu',
]
COMPILER = {
    'optimizer': [
        'adam'
    ],
    'loss': [
        'mean_squared_error',
        'mean_absolute_error',
        'mean_absolute_percentage_error',
        'mean_squared_logarithmic_error',
        'cosine_similarity',
        'huber',
        'log_cosh',

    ],
    'metrics': [
        'accuracy'

    ]
}
ACTIVE_LEN = len(ACTIVATION) - 1
TICKERS = ['NIO', 'TSLA', 'BABA', 'XPEV', 'FB', 'AAPL']
tickers_length = len(TICKERS) - 1


def generate_layers(units):
    length = randint(4, 10)
    layers = []
    for _ in range(length):
        layer = LAYERS[randint(0, 2)]
        activation = ACTIVATION[randint(0, ACTIVE_LEN)]
        if layer == 'Dropout':
            layer_args = {'units': 0.2, 'activation': activation}
        else:
            layer_args = {'units': (next_units := randint(1, units)), 'activation': activation}
            units = next_units
        layers.append({layer: layer_args})
    return layers


def _units(units):
    return units // 2 if bool(getrandbits(1)) else int(units * 0.75)


def generate_single_random_layer(index, units):
    return {(layer := 'Dense') if index == 0 else (layer := LAYERS[randint(0, 2)]): {
        'units': (units := _units(units)) if layer != 'Dropout' else 0.2,
        'activation': ACTIVATION[randint(0, ACTIVE_LEN)]}}, units


def generate_layers_from_father(father):
    layers_list = combinations_with_father_list((layers := father[3]))
    units = father[1]
    layers_list2 = []

    for index_layers, (should, layers_key) in enumerate(zip(layers_list, layers)):
        layers_list2.append([
            (units := generate_single_random_layer(index, units if type(units) is int else units[1]))[0]
            if not layer else layers[index] for index, layer in enumerate(should)
        ])
        layers_list2.append(generate_layers(units if type(units) is int else units[1]))
        if units < 4:
            break

    return layers_list2


def generate_compile():
    return {key: COMPILER[key][randint(0, len(COMPILER[key]) - 1)] if key != 'metrics' else list({
        COMPILER[key][randint(0, len(COMPILER[key]) - 1)] for _ in range(randint(1, 3))}) for key in COMPILER}


def settings_generator(children=None):
    if children is None:
        settings = {
            'epochs': randint(10, 30),
            'units': (units := randint(12, 70)),
            'prediction_days': randint(5, 90),
            'layers': generate_layers(units),
            'compiler': generate_compile()

        }
    else:
        print(children)
        units = True
        settings = [{
            'epochs': randint(10, 30) if not child[0] else child[0],
            'units': (units := randint(12, 70)) if not child[1] else child[1],
            'prediction_days': randint(5, 90) if not child[2] else child[2],
            'layers': generate_layers(units if type(units) is int else child[1]) if not child[3] else child[3],
            # TODO  generate from those more children
            'compiler': generate_compile() if not child[4] else child[4]

        } for child in children]
        print(children)
        secondary_settings = [
            {
                'epochs': children[-1][0],
                'units': children[-1][1],
                'prediction_days': children[-1][2],
                'layers': layer,
                'compiler': children[-1][4]

            }
            for layer in generate_layers_from_father(children[-1])]
        settings.extend(secondary_settings)

    return settings


def generate_settings_using_father_dna(father):
    children = combinations_with_father_list(father)
    children = list(
        [value if set_value else set_value for value, set_value in
         zip(father.values() if type(father) is dict else father, child)]
        for child in children
    )
    return settings_generator(children)


def combinations_with_father_list(father):
    children = list(
        itertools.combinations_with_replacement([True, False], len(father))
    )
    second = itertools.combinations_with_replacement([False, True], len(father))
    children.extend(second)
    children.sort()
    children = list(children for children, _ in itertools.groupby(children))
    return children


def generate_classifier_ai_stocks_objects(children: list):
    children_ratio_dict = {}
    for index, child in enumerate(children):
        epochs, units, prediction_days, layers, compiler = child.values()
        print(epochs, units, prediction_days, layers, compiler, ':)', sep=', ')
        child_classifier_obj = ClassifierAi(TICKERS[randint(0, tickers_length)],
                                            epochs=epochs,
                                            units=units,
                                            prediction_days=prediction_days,
                                            model_building_blocks={'layers': layers, 'compiler': compiler})
        ratio = float(
            (pr := str(child_classifier_obj.test_model_and_return_accuracy_ratio()[-1]))[0: 7 if len(pr) > 6 else -1])
        children_ratio_dict[ratio] = {'epochs': epochs, 'units': units,
                                      'prediction_days': prediction_days, 'layers': layers,
                                      'compiler': compiler}

    return children_ratio_dict


def _main(settings):
    """
    :returns best of all kids and makes it the next parent
    """
    children = generate_settings_using_father_dna(settings)
    classifier_ai_stock_objects = generate_classifier_ai_stocks_objects(children)
    return classifier_ai_stock_objects[
               (best_ratio := max(classifier_ai_stock_objects.keys()))
           ], best_ratio


def main():
    father, last_ratio = Cm.best_settings()
    print(father, last_ratio)
    import time
    time.sleep(2)
    for _ in range(100):
        father, best_ratio = _main(father)
        if best_ratio != last_ratio or str(best_ratio) != str(last_ratio):
            # Avoid error by putting two keys with same value
            data = {best_ratio: father}
            Cm.write_in_json('./settings_for_ai/parameters_status.json', data)
            last_ratio = best_ratio


if __name__ == '__main__':
    main()

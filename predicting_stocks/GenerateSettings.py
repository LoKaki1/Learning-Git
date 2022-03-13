from random import randint
import itertools
from ClassifierStocksAiClass import ClassifierAi


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
    'deserialize', 'elu', 'exponential', 'gelu', 'get',
    'hard_sigmoid', 'linear', 'relu', 'selu', 'serialize', 'sigmoid',
    'softmax', 'softplus', 'softsign', 'swish', 'tanh'
]
COMPILER = {
    'optimizer': [
        'sgd', 'rmsprop', 'adam', 'adadelta', 'adagrad', 'adamax', 'nadam', 'ftrl'
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
        'AUC',
        'Precision',
        'Recall',
        'TruePositives',
        'TrueNegatives',
        'FalsePositives',
        'FalseNegatives',
        'PrecisionAtRecall',
        'SensitivityAtSpecificity',
        'SpecificityAtSensitivity',

    ]
}
ACTIVE_LEN = len(ACTIVATION) - 1
TICKER = 'NIO'


def generate_layers(units):
    length = randint(4, 10)
    layers = {}
    for _ in range(length):
        layer = LAYERS[randint(0, 2)]
        activation = ACTIVATION[randint(0, ACTIVE_LEN)]
        if layer == 'Dropout':
            layer_args = {'units': 0.2, 'activation': activation}
        else:
            layer_args = {'units': (next_units := randint(10, units)), 'activation': activation}
            units = next_units
        layers[layer] = layer_args
    return layers


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
            'compile': generate_compile()

        }
    else:
        units = True
        settings = [{
            'epochs': randint(10, 30) if not child[0] else child[0],
            'units': (units := randint(12, 70)) if not child[1] else child[1],
            'prediction_days': randint(5, 90) if not child[2] else child[2],
            'layers': generate_layers(units if type(units) is int else child[1]) if not child[3] else child[3],
            # TODO  generate from those more children
            'compile': generate_compile() if not child[4] else child[4]

        } for child in children]

    return settings


def generate_settings_using_father_dna(father):
    children = list(
        itertools.combinations_with_replacement([True, False], len(father))
    )
    second = itertools.combinations_with_replacement([False, True], len(father))
    children.extend(second)
    children.sort()
    children = list(children for children, _ in itertools.groupby(children))
    children = list(
                    [value if set_value else set_value for value, set_value in zip(father.values(), child)]
                    for child in children
                    )
    return settings_generator(children)


def generate_classifier_ai_stocks_objects(children: list):
    children_ratio_dict = {}
    for index, child in enumerate(children):
        epochs, units, prediction_days, layers, compiler = child.values()
        print(epochs, units, prediction_days, layers, compiler )
        child_classifier_obj = ClassifierAi(TICKER,
                                            epochs=epochs,
                                            units=units,
                                            prediction_days=prediction_days,
                                            model_building_blocks={'layers': layers, 'compiler': compiler})
        children_ratio_dict[index] = ([epochs, units, prediction_days, layers, compiler],
                                      child_classifier_obj.test_model_and_return_accuracy_ratio())
    return children_ratio_dict


def _main(settings):
    children = generate_settings_using_father_dna(settings)
    classifier_ai_stock_objects = generate_classifier_ai_stocks_objects(children)

    'for loop on each of them test model, and return dictionary {id: [settings, ratio]}'


def main():
    settings = ['__generate_settings_from_file__ if is not none else None']
    units = 50
    father = settings_generator()
    _main(father)


main()

from flask import Flask, request
from predicting_stocks.ClassifierStocksAiClass import ClassifierAi
from fla
import json


app = Flask(__name__)
cors = CORS(app, resources={r"/foo": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/login', methods=['POST'])
def login():
    user, password, database = (j := request.get_json())['user'],  j['password'], _open_json('sami.json')
    return json.dumps({
        'message': 'connected :)' if user in database and database[user] == password else 'not connected'
    })


@app.route('/sign')
def sign_up():
    pass


@app.route('/predict', methods=['POST'])
def predict_stock():
    print(request.get_json(), request.headers)
    ticker, epochs, units, prediction_days, prediction_day,  = json_contains(
                                                               (data := request.get_json()), 'ticker', 'NIO'),\
                                                               json_contains(data, 'epochs'),\
                                                               json_contains(data, 'units'),\
                                                               json_contains(data, 'prediction_days'),\
                                                               json_contains(data, 'prediction_day')

    stock_object = ClassifierAi(ticker,
                                epochs=epochs,
                                units=units,
                                prediction_days=prediction_days,
                                prediction_day=prediction_day)
    return "".join([f'The price of {ticker} - ', str(stock_object.predict_stock_price_at_specific_day()[-1][-1])])


def _open_json(path):
    with open(path, 'r') as file:
        json_data = json.loads(file.read())
    return json_data


def json_contains(data, key, value=None):
    return data[key] if key in data else value


if __name__ == '__main__':
    app.run(host='localhost', )

from flask import Flask, request
from predicting_stocks.ClassifierStocksAiClass import ClassifierAi
import predicting_stocks.Common as Cm
from flask_cors import CORS
import json


app = Flask(__name__)
cors = CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'application/json'


@app.route('/login', methods=['POST'])
def login():
    user, password, database = (j := request.get_json())['user'], j['password'], Cm.open_json('sami.json')
    return json.dumps({
        'message': 'connected :)' if user in database and database[user] == password else 'not connected'
    })


@app.route('/sign')
def sign_up():
    pass


@app.route('/predict', methods=['POST'])
def predict_stock():
    ticker, epochs, units, prediction_days, prediction_day, = (data := dict(
        request.get_json())).get(
        'ticker', 'NIO'), data.get(
        'epochs'), data.get('units'), data.get(
        'prediction_days'), data.get(
        'prediction_day')
    json_object = Cm.open_json('../api/database.json')
    if isinstance(today := Cm.handle_with_time(ticker, json_object), float):
        pass
    else:
        stock_object = ClassifierAi(ticker,
                                    epochs=epochs,
                                    units=units,
                                    prediction_days=prediction_days,
                                    prediction_day=prediction_day)

        Cm.save_in_data_base(ticker,
                             str(stock_object.predict_stock_price_at_specific_day()[-1][-1]),
                             stock_object.get_settings(),
                             today,
                             Cm.get_last_id(json_object))
    return _create_watchlist()


@app.route('/watchlist', methods=['GET'])
def watchlist():
    return _create_watchlist()


def _create_watchlist():
    data = Cm.open_json('../api/database.json')
    recreate_data = []
    for i in data:
        data[i]['ticker'] = i
        recreate_data.append(data[i])
    return json.dumps(recreate_data)


@app.route('/current', methods=['POST'])
def current_price():
    pass


if __name__ == '__main__':
    app.run(host='localhost', )

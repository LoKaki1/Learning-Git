from flask import Flask, request
from predicting_stocks.ClassifierStocksAiClass import ClassifierAi
import predicting_stocks.Common as Cm
from flask_cors import CORS
import json
import datetime as dt

app = Flask(__name__)
cors = CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'application/json'


@app.route('/login', methods=['POST'])
def login():
    user, password, database = (j := request.get_json())['user'], j['password'], _open_json('sami.json')
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

    if isinstance(today := handle_with_time(ticker), float):
        return "".join([f'The price of {ticker} - ', str(today)])
    stock_object = ClassifierAi(ticker,
                                epochs=epochs,
                                units=units,
                                prediction_days=prediction_days,
                                prediction_day=prediction_day)

    save_in_data_base(ticker,
                      stock_price := str(stock_object.predict_stock_price_at_specific_day()[-1][-1])[0:6:],
                      stock_object.get_settings(),
                      today)
    return "".join([f'The price of {ticker} - ', str(stock_price)])


def handle_with_time(ticker):
    today_str = (today  := dt.datetime.now()).strftime('%d-%b-%Y')
    ticker_last_date = Cm.open_json_file('database.json')
    if ticker not in ticker_last_date:
        return today_str
    ticker_last_date = ticker_last_date[ticker]
    return today_str if dt.datetime.strptime(
        ticker_last_date[
            'date'], '%d-%b-%Y') < dt.datetime.strptime(today_str, '%d-%b-%Y') else float(ticker_last_date['price'])


def save_in_data_base(ticker, price, settings, date):
    data = {
        ticker: {
            "price": price,
            "settings": settings,
            "date": date
        }
    }
    Cm.write_in_json_file('database.json', data, ticker)


def _open_json(path):
    with open(path, 'r') as file:
        json_data = json.loads(file.read())
    return json_data


def json_contains(data, key, value=None):
    return data[key] if key in data else value


if __name__ == '__main__':
    app.run(host='localhost', )

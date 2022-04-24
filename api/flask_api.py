import datetime as dt
from flask import Flask, request
from predicting_stocks.ClassifierStocksAiClass import ClassifierAi, END
import predicting_stocks.Common as Cm
from flask_cors import CORS
import json
from Trading.ScannerInIneractive import get_most_gain_scanner

app = Flask(__name__)
cors = CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'application/json'
user_database_path = '../api/Databases/sami.json'
user_token_wrong = json.dumps({'data': 'lost connection try to relog in'})


def _get_user_password_and_data_base():
    return (data := dict(request.get_json())).get('user'), data.get('password'), Cm.open_json(user_database_path)


@app.route('/register', methods=['POST'])
def register():
    """
    function to register
    json post = {user: "username", password: "password"}
    """
    username, password, user_database = _get_user_password_and_data_base()
    if username in user_database['users']:
        return json.dumps({'data': 'User with this username already exists'})
    if username is None or password is None or password == username:
        return json.dumps({'data': 'Not valid user name or password'})
    user_database['users'][username] = password
    Cm.write_in_json(path=user_database_path, data=user_database)
    return json.dumps({'data': 'User added to database'})


@app.route('/login', methods=['POST'])
def login():
    """
    function to log in
    """
    username, password, user_database = _get_user_password_and_data_base()
    if username not in user_database['users'] or user_database['users'][username] != password:
        return json.dumps({'data': 'Username or password not valid'})
    token = Cm.generate_tokens(username)
    return json.dumps({'data': 'User connected :)', 'token': token})


@app.route('/predict', methods=['POST'])
@Cm.token_checking
def predict_stock():
    ticker, epochs, units, prediction_days, prediction_day, = (data := dict(
        request.get_json())).get(
        'ticker', 'NIO'), data.get(
        'epochs'), data.get('units'), data.get(
        'prediction_days'), data.get(
        'prediction_day')
    json_object = Cm.open_json('Databases/database.json')

    stock_object = ClassifierAi(ticker,
                                epochs=epochs,
                                units=units,
                                prediction_days=prediction_days,
                                prediction_day=prediction_day,
                                load_model_from_local=False,
                                )
    date = Cm.handle_with_time(ticker, json_object)
    ticker_settings, settings_not_found = handle_settings(stock_object, json_object, ticker)
    if type(date) is not float or settings_not_found:
        price = stock_object.predict_stock_price_at_specific_day(True)
        predicted_price = str(price)
        Cm.save_in_data_base(ticker,
                             predicted_price,
                             ticker_settings,
                             dt.datetime.now().strftime('%Y-%m-%d'),
                             Cm.get_last_id(json_object)
                             )
    return _create_watchlist()


def handle_settings(stock_object, json_object, ticker):
    settings = stock_object.get_settings()
    ticker_settings = list(json_object[ticker]['settings']) if ticker in json_object else []
    if settings not in ticker_settings:
        ticker_settings.append(settings)
        return ticker_settings, True
    return ticker_settings, False


@app.route('/watchlist', methods=['POST'])
@Cm.token_checking
def watchlist():
    return _create_watchlist()


def _create_watchlist():
    data = Cm.open_json('Databases/database.json')
    recreate_data = []
    for i in data:
        data[i]['ticker'] = i
        recreate_data.append(data[i])
    return json.dumps(recreate_data)


@app.route('/prices/daily', methods=['POST'])
@Cm.token_checking
def current_price():
    ticker = dict(request.get_json()).get('ticker')
    data = Cm.get_historical_data(ticker, start := dt.datetime(year=2021, month=1, day=1).strftime('%Y-%m-%d'), END)
    data = Cm.iterate_data(data)
    dates = Cm.generate_dates_between_dates(start, END)
    data = [{'x': date.strftime(
        '%Y-%m-%d'), 'y': [
        float(str(
            price)[0: 5])
        for price in prices]}
        for date, prices in zip(dates, data) if data[-1] != prices]
    return json.dumps(data)


@app.route('/prices/interday', methods=['POST'])
@Cm.token_checking
def interday():
    (ticker, period, interval) = (request.get_json().get('ticker'),
                                  request.get_json().get('period', 1),
                                  request.get_json().get('interval', '1m'))
    data = Cm.no_iteration_interday_with_yahoo(ticker, period, interval)
    data = [
        {
            'x': key,
            'y': [float(str(value)[0: 5]) for value in values]
        }
        for key, values in data.items()]
    return json.dumps(data)


@app.route('/scanners/most_gainers', methods=['POST'])
@Cm.token_checking
def get_scanner():
    (scanner, scanner_args) = request.get_json().get('scanner'), request.get_json().get('scanner_args')
    return json.dumps(get_most_gain_scanner(scanner=scanner, scanner_args=scanner_args))

if __name__ == '__main__':
    app.run(host='localhost', )

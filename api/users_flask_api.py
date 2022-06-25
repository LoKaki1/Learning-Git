"""

This script the flask api to handle with user authentication, database, and data managements

    each request should look like this -
     {
        data: {}
        headers: {token, user}
    }


"""
import json

from flask import Flask, request
from flask_cors import CORS
from functools import wraps
import predicting_stocks.Common as cm
from predicting_stocks.ClassifierStocksAiClass import ClassifierAi

app = Flask(__name__)
cors = CORS(app, support_credentials=True)
app.config['CORS_HEADERS'] = 'application/json'
DECORATOR_ARGS = ['url', 'methods']
USER_DATABASE = 'Databases/sami.json'


def _login_register_helper(client_request):
    if not cm.list_in_list(['user', 'password'], client_request):
        return 'request does not contain user and password keys', 'bad request 400', False
    return (client_request, cm.open_json(USER_DATABASE)), '200 ok', True


def checklist(**options):
    client_headers = request.headers
    client_args = request.args
    client_request = request.get_json()
    if 'POST' in request.method and 'data' not in client_request:
        return 'The request doesn\'t contains "data" key', 'bad request 400', False

    if 'token' in options:
        f" Here check for token authentication return error 405  because no authentication"
        return cm.token_authentication(client_headers, client_request, client_args)

    if 'auth' in options:
        return _login_register_helper(client_request['data'])

    return 'What the hell are you looking for m8', 'Not Found 404', False


def predict_stocks_and_save(ticker, headers, **client_args):
    stock_object = ClassifierAi(ticker, epochs=12, prediction_days=60, prediction_day=6, units=100,
                                daily=True if 'interday' not in client_args else False, load_model_from_local=False,
                                use_best_found_settings=False,
                                source='yahoo')
    predicted_data = stock_object.predict_price(**client_args)
    predicted_data.update({'id': cm.generate_uniq_id()})
    cm.add_to_database(headers['user'], {ticker: predicted_data}, 'watchlist')
    return predicted_data


def decorator(**options):
    def inner_function(function):

        @wraps(function)
        def wrapper():
            check = checklist(**options)
            if not check[-1]:
                return json.dumps({'data': check[0], 'status': check[1]})
            data, status = function(check[0])
            return json.dumps({'data': data, 'status': status})
        return wrapper
    return inner_function


@app.route('/register', methods=['POST'])
@decorator(auth=True)
def register(register_request, ):
    """
        Handles with register wants to get {"user": string, "password": string}
    """
    register_request, users = register_request
    if (user := register_request['user']) in users:
        return f'user with the username {user} already exists', 409
    cm.write_in_json(USER_DATABASE, {user: register_request['password']})
    return f'User {user} added to database 😍', 200


@app.route('/login', methods=['POST'])
@decorator(auth=True)
def login(login_request):
    """
        Handles with login wants to get {"user": string, "password": string}
        return token with that the server and client talks with each other
    """

    login_request, users = login_request
    if (user := login_request['user']) not in users or users[user] != login_request['password']:
        return f'User or password are not correct', 403
    return {'token': cm.generate_tokens(login_request['user'])}, 200


@app.route('/predict', methods=['GET'])
@decorator(token=True)
def predict(keys):
    """
        Function to predict specific stock -
            GET request, HEADERS = {'token':}, ARGS = {ticker: }
            url = /predict?ticker=name&ratio=?  (ratio - potential argument)
        :return -
            {
                data: {
                        ticker:
                        predicted_price:
                        last_price:
                        change:
                        presntage_change:
                    }
            }
    """
    # Get args, headers and request if post
    args = keys['args']

    headers = keys['headers']
    ticker, client_args = args.get('ticker', False), dict(args)
    client_args.pop('ticker')

    if not ticker:
        return 'No ticker in args', 'bad request 400'
    predicted_data = predict_stocks_and_save(ticker, headers, **client_args)

    return predicted_data, 200


@app.route('/predict/watchlist', methods=['GET'])
@decorator(token=True)
def predict_watchlist(keys):
    user = (headers := keys['headers'])['user']
    client_args = keys['args']
    user_database = cm.open_json('../api/Databases/user_database.json')
    if user not in user_database:
        return "No tickers in watchlist", "bad request 400"
    watchlist = {ticker: predict_stocks_and_save(ticker, headers, **client_args)
                 for ticker in user_database[user]['watchlist']}

    return watchlist, 200


@app.route('/prices/daily', methods=['GET'])
@decorator(token=True)
def get_daily_data(keys):
    user, ticker, client_args = (headers := keys['headers'])['user'], (args := dict(keys['args'])).pop('ticker'), args
    try:
        start = client_args.get('start', cm.get_user_start_day(user, 'start'))
        cm.add_to_database(user, data={'start': start}, key='dates')
        return cm.histrical_data_json_format(ticker, start), 200
    except [ValueError, KeyError]:
        return "Something went wrong try predict stock price and then ask for its graph", "Bad Request 400"


@app.route('/prices/interday', methods=['GET'])
@decorator(token=True)
def get_interday_data(keys):
    user, ticker, client_args = (headers := keys['headers'])['user'], (args := dict(keys['args'])).pop('ticker'), args
    try:
        period, interval = client_args.get('period', cm.get_user_start_day(user, 'period')),\
                           client_args.get('interval', cm.get_user_start_day(user, 'interval'))
        cm.add_to_database(user, data={'period': period, 'interval': interval}, key='dates')
        return cm.interday_data_json_format(ticker, period, interval), 200
    except (ValueError, KeyError, AttributeError) as e:

        return f"Something went wrong try predict stock price and then ask for its graph {e}", "Bad Request 400"


@app.route('/active/watchlist', methods=['POST'])
@decorator(token=True)
def active_watchlist(keys):
    """
        user_request = {data: [tickers_to_active_watchlist]}
    """
    user, user_request = keys['header']['user'],  keys['request']


if __name__ == '__main__':
    app.run()

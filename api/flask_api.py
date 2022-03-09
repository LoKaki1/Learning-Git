
import datetime as dt
from flask import Flask, request
from predicting_stocks.ClassifierStocksAiClass import ClassifierAi, END
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
    # if isinstance(today := Cm.handle_with_time(
    #         ticker,
    #         json_object,
    #         epochs,
    #         units,
    #         prediction_days,
    #         prediction_day), float):
    #     pass

    stock_object = ClassifierAi(ticker,
                                epochs=epochs,
                                units=units,
                                prediction_days=prediction_days,
                                prediction_day=prediction_day,
                                load_model_from_local=True,
                                )
    date = Cm.handle_with_time(ticker, json_object)
    ticker_settings, settings_not_found = handle_settings(stock_object, json_object, ticker)
    if type(date) is not float or settings_not_found:
        predicted_price = str(stock_object.predict_stock_price_at_specific_day()[-1][-1])
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


@app.route('/prices', methods=['POST'])
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
    print(data)
    return json.dumps(data)


if __name__ == '__main__':
    app.run(host='localhost', )
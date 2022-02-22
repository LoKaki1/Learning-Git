from flask import Flask, request
from predicting_stocks.ClassifierStocksAiClass import ClassifierAi
import json


app = Flask(__name__)


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
    ticker = (data := request.get_json())['ticker']
    stock_object = ClassifierAi(ticker,)
    return "".join([f'The price of {ticker}', str(stock_object.predict_stock_price_at_specific_day()[-1][-1])])


def _open_json(path):
    with open(path, 'r') as file:
        json_data = json.loads(file.read())
    return json_data


if __name__ == '__main__':
    app.run()

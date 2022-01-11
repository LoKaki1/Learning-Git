""" This script buys and sells stocks according to what robot says """

import time
import yfinance as yf

STOCKS_TO_PREDICT = ['NIO', 'XPEV', 'LI', 'TTOO', 'RIOT', 'TSLA']
STOCKS_PREDICTED = {'NIO': [42], 'XPEV': [43], 'LI': [30]}
RISK = 1.015
PROFIT = 1.03


def buy(variables):
    """  buys a stock in a price """
    ticker, predicted_price, actions, real_price, profit, risk = variables
    if ticker not in actions or not actions[ticker][-1]:
        if predicted_price >= real_price * profit:
            actions[ticker] = ['BUY', real_price, None, True]
            return -real_price

    elif actions[ticker][0] == 'SELL' and actions[ticker][3]:
        if real_price >= actions[ticker][2] * risk:
            actions[ticker][1] = real_price
            actions[ticker][3] = False
            return - real_price
        elif real_price <= actions[ticker][2] * (2 - profit):
            actions[ticker][1] = real_price
            actions[ticker][3] = False
            return -real_price
    return 0


def sell(variables):
    """   sells a stock in a price """
    ticker, predicted_price, actions, real_price, profit, risk = variables
    if ticker not in actions or not actions[ticker][-1]:
        '''  Means if predicted price is lower then real price I want you to go short  
             looks like this    predicted_price = 20  real_price = 22    profit = 1.03
             if 20 <= 22 * (2-1.03) short func
        '''
        if predicted_price <= real_price * (2 - profit):
            actions[ticker] = ['SELL', None, real_price, True]
            return real_price

    elif actions[ticker][3] and actions[ticker][0] == 'BUY':
        ''' No need to split those two but it makes
         it more simple to understand it because the first one is if the buy has gotten to stop loss

         means if price <= risk price         
         and the other one is if take profit 
         if price >= profit price 

         1 : 3 default ratio
         '''
        if real_price <= actions[ticker][1] * (2 - risk):
            actions[ticker][2] = real_price
            actions[ticker][3] = False
            return -real_price

        elif real_price >= actions[ticker][1] * profit:
            actions[ticker][2] = real_price
            actions[ticker][3] = False
            return real_price
    return 0


def stop_profit(variables):
    """
    stop loss or take_profit from a stock if touched a specific price
    price:  the price it does the transaction
    action: whether it should buy or sell

    """
    pass


def get_last_price(ticker: str):
    """
    Return last price of a stock
    """
    try:
        return yf.download(tickers=ticker, period='1d', interval='1m')['Close'][-1]
    except IndexError:
        time.sleep(30)
        return yf.download(tickers=ticker, period='1d', interval='1m')['Close'][-1]
    # return random.randrange(0, 100)


def predict(ticker):
    """
    Returns a predict price from a ticker of today (Want to keep the screen clean as f#%$)
    """
    return predict_stocks_avg(ticker, average=5)


def predict_dict():
    print('somethin')
    return dict((i, [predict(i), get_last_price(i)]) for i in STOCKS_TO_PREDICT)


def stocks_worth_trading(stocks_prediction, profit=PROFIT, ):
    """

    Returns a dict for all stocks that has good

    """
    return dict((ticker, stocks_prediction[ticker])
                for ticker in stocks_prediction if
                max(stocks_prediction[ticker][0] / stocks_prediction[ticker][1],
                    stocks_prediction[ticker][1] / stocks_prediction[ticker][0]) >= get_last_price(ticker) * profit
                or max(stocks_prediction[ticker][0] / stocks_prediction[ticker][1],
                    stocks_prediction[ticker][1] / stocks_prediction[ticker][0]) <= get_last_price(ticker) * (2 - profit))


def update_file(actions, money_achieved, path='documentation.txt'):
    """
    Updates the file for trading documentation whether it bought sold shorted or covered and which prices
    """

    actions_data = ''
    for i in range(len(actions)):
        for ticker in actions[i]:
            print(actions[i][ticker])
            actions_data += f'Trade {i}: ' \
                            'Ticker - ' + ticker + \
                            '  Bought in price - ' + \
                            f'{actions[i][ticker][1]}' + \
                            f'  Sold in price - {actions[i][ticker][2]} \n'

    data = f'money - {money_achieved} \n' \
           f' \n' \
           f' \n' \
           f'{actions_data}'
    with open(path, 'w') as file:
        file.write(data)
        file.close()


def money(actions):
    """

    Calculate how much money m.r robot achieved or lost

    """
    return sum(actions[i][2] - actions[i][1] for i in actions if not actions[i][3])


def main():
    stocks_prediction = predict_dict() if STOCKS_PREDICTED == {} else dict(
        (i, [STOCKS_PREDICTED[i][0], get_last_price(i)]) for i in STOCKS_PREDICTED)

    stocks_to_trade = stocks_worth_trading(stocks_prediction)
    actions = {}
    "actions = {ticker: [type, bought_price, sold_price, status]}"
    money_achieved = 0
    '''  While loop because it works all day long '''
    actions_list = []
    actions_copy = actions.copy()
    while True:
        for index, i in enumerate(stocks_to_trade):
            list_variables = [i, stocks_to_trade[i][0], actions, get_last_price(i), PROFIT, RISK, ]
            print(f'---------- Trade {index}-----------')
            money_achieved += buy(list_variables)
            money_achieved += sell(list_variables)
            print(actions, actions_copy)
            if actions_copy != actions:
                actions_copy = actions.copy()
                actions_list.append(actions_copy)
                update_file(actions_list, money_achieved)


main()
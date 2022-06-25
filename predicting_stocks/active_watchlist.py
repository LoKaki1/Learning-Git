"""

class for automate trading using the prediction from the ClassifierStocksAi prediction,

    Way it's going to work -
        1. getting active watchlist
        2. For each stock do the following -
            a. predict price in **kwargs settings,
            b. get the right now price
            c. if there's a chance to make 3% buy or sell accordingly
            d. risk 1 %
            e. put them in json object that contains all stocks data
        3. For each active stock (active stock means in protfolio);
            a. check if it got to stop loss
            b. check for take profit
            c. remove from active,
            d. add loss or profit to total_revenue

"""
import predicting_stocks.Common as cm
from predicting_stocks.ClassifierStocksAiClass import ClassifierAi

ACTIVE_WATCHLIST_PATH = '../api/Databases/a.json'


def helper(current_price, entry_price):
    return {
        'left': current_price,
        'revenue': current_price - entry_price
    }


class ActiveWatchlist:
    def __init__(self, user, **kwargs):
        self.user = user
        self.stocks_kwargs = kwargs
        self.watchlist = []
        self.active_watchlist = {'active_stocks': [
            # {stock_name, buy | short: price, predicted_price, loss, profit, sell | cover: price}
        ], 'close_stocks': [],
            "revenue": 0}
        self.revenue = 0

    def main_loop(self):
        # Reason for this not to be in the init function is because I want in each iteration to get the new stocks
        # that the user added if added
        self.watchlist = cm.open_json(cm.DATABASE_PATH)[self.user]['active_watchlist']
        last_active_watclist = cm.open_json(ACTIVE_WATCHLIST_PATH)
        self.revenue = last_active_watclist.get('revenue', 0)
        for stock in self.watchlist:
            stock_object = ClassifierAi(stock, **self.stocks_kwargs)
            if (last_stock_data := cm.found_in_list_of_dict(last_active_watclist['active_stocks'], 'stock', stock))[0]:
                if (last_stock_data := (last_stock_data[1]))['state']:
                    self.close_position(stock, stock_object, last_stock_data)
                else:
                    self.active_watchlist['active_stocks'].append(last_stock_data)
                    continue
            self.open_position(stock, stock_object)
        self.active_watchlist['revenue'] = last_active_watclist.get('revenue', 0) + self.revenue
        cm.write_json(ACTIVE_WATCHLIST_PATH, self.active_watchlist)

    def close_position(self, stock, stock_object, last_stock_data):
        """
            closes open position if current price is close to loss or profit, revenue = buy | short - sell | cover
            :return -
                {stock_name, type[buy, short], revenue, state: if close, entry:, left:}
        """
        new_stock_data = {}
        price = stock_object.live_last_price()
        if (transaction := last_stock_data['type']) == 'buy':
            if price >= last_stock_data['profit'] or price <= last_stock_data['loss']:
                new_stock_data.update(helper(price, last_stock_data['entry']))
        elif transaction == 'short':
            if price <= last_stock_data['profit'] or price >= last_stock_data['loss']:
                new_stock_data.update(helper(price, last_stock_data['entry']))
        else:
            return
        new_stock_data['name'] = stock
        self.revenue += new_stock_data['revenue']
        self.active_watchlist['active_stocks'].append(new_stock_data)

    def open_position(self, stock: str, stock_object: ClassifierAi):
        """
            opens position if predicts more than 3% change_presntage buy stock and risk 1% below,
            or if predicts less than -3% change_presntage short stock and risk 1% above
        """
        predicted_list = stock_object.predict_price()
        new_stock_data = {}
        if predicted_list['change_presentage'] > 3:
            new_stock_data['type'] = 'buy'
            new_stock_data['entry'] = last_price = predicted_list['last_price']
            new_stock_data['profit'] = last_price * 1.03
            new_stock_data['loss'] = last_price * 0.98
        elif predicted_list['change_presentage'] < -3:
            new_stock_data['type'] = 'short'
            new_stock_data['entry'] = last_price = predicted_list['last_price']
            new_stock_data['profit'] = last_price * 0.97
            new_stock_data['loss'] = last_price * 1.015
        else:
            return
        new_stock_data['name'] = stock
        self.active_watchlist['active_stocks'].append(new_stock_data)


active = ActiveWatchlist('a123', daily=True, source='yahoo', other=2, prediction_day=15)
while True:
    active.main_loop()

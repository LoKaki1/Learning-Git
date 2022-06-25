"""
    active_watchlist - [...]
    Wanted Data -
        JSON -
        first -
            ticker_name: {
                predicted_price, date (%Y-%m-%d %H:%M)
            }
        for each first -
            if predicted_price > 5% from current_price - long
            if predicted_price < -5% from current_price - short
        second -
            ticker_name: {
                 predicted_price, entry_price, type
            }
            if one of them then remove from active_watchlist - and add to protfolio
    For protfolio -
        if current >= predicted_price or current <= entry_price * 0.97- sell
        if current >= predicted_price or current >= entry_price * 1.02 - cover
        remove from protfolio  and back for watchlist...

"""
import time

from ClassifierStocksAiClass import ClassifierAi
import datetime as dt
import Common as cm

ACTIVE = '../api/Databases/active.json'
PROTFOLIO = '../api/Databases/protfolio.json'


def active_object(swing_trading_object):
    return {
        swing_trading_object.ticker: {
            "predicted_price": swing_trading_object.predicted_price,
            "date": dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
            "current_price": swing_trading_object.current_price
        }
    }


def protfolio_object(swing_trading_object):
    return {
        swing_trading_object.ticker: {
            "predicted_price": swing_trading_object.predicted_price,
            "date": dt.datetime.now().strftime('%Y-%m-%d %H:%M'),
            "current_price": (cr := swing_trading_object.current_price),
            "type": (tp := swing_trading_object.trade_type),
            "entry_price": (ep := swing_trading_object.entry_price),
            "money": cr - ep if tp == 'long' else ep - cr
        }
    }


def remove_active_and_add_to_protfolio(swing_trading_object):
    cm.remove_key_from_json(ACTIVE, swing_trading_object.ticker)
    cm.write_json(PROTFOLIO, protfolio_object(swing_trading_object))


def remove_protfolio_and_add_to_active(swing_trading_object):
    cm.remove_key_from_json(PROTFOLIO, swing_trading_object.ticker)
    cm.write_json(ACTIVE, active_object(swing_trading_object))


class SwingTradingActive(ClassifierAi):
    """
    This class is for one stock -
        1. Get Stock Object
        2. predict price
        3. get current price
        4. check the above
        5. move to profotfolio if needed
    """

    def __init__(self, ticker, **kwargs):
        super().__init__(ticker, **kwargs)
        self.predicted_price = None
        self.current_price = self.live_last_price()
        self.entry_price = None
        self.trade_type = None

    # def active_or_protfolio(self):
    #     if self._checklist():
    #         self.remove_from_active()
    #         self.add_to_protfolio
    #

    def _checklist(self):
        self.predicted_price = self.predict_stock_price_at_specific_day()
        if self.predicted_price >= 1.05 * self.current_price:
            self.trade_type = 'long'
            self.entry_price = self.current_price
            return True

        if self.predicted_price <= 0.95 * self.current_price:
            self.trade_type = 'short'
            return True
        return False

    def add_to_list(self):
        cm.write_in_json(ACTIVE, active_object(self))
        if not self._checklist():
            return
        remove_active_and_add_to_protfolio(self)


class SwingTradingProtfolio(SwingTradingActive):
    def __init__(self, ticker, pr, **kwargs):
        super().__init__(ticker, **kwargs)
        self.money, self.trade_type, self.entry_price = pr['money'], pr['type'], pr['entry_type']

    def _checklist(self):
        return True if (self.trade_type == 'long' and (self.current_price >= self.predicted_price or
                                                       self.current_price <= self.entry_price * 0.97)) or \
                       (self.trade_type == 'short' and (self.current_price >= self.predicted_price or
                                                        self.current_price >= self.entry_price * 1.02)) \
            else False

    def add_to_list(self):
        if not self._checklist():
            return

        remove_protfolio_and_add_to_active(self)
        return self.money


def main():
    actives = ['NIO', 'XPEV', 'RIOT', 'MARA', 'LI']
    kwargs = {}
    money = 0
    while True:
        for active in actives:
            start_swing = SwingTradingActive(active, **kwargs)
            start_swing.add_to_list()
            time.sleep(5)
        protfolios = cm.open_json(PROTFOLIO)
        for protfolio, protfolio_items in protfolios.items():
            stop_swing = SwingTradingProtfolio(protfolio, protfolio_items, **kwargs)
            money += stop_swing.add_to_list()
        actives = cm.open_json(ACTIVE)
        print(money)


if __name__ == '__main__':
    main()

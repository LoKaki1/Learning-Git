from webbrowser import get

from ibapi import wrapper
from ibapi.client import EClient
from ibapi.utils import iswrapper  # just for decorator
from ibapi.common import *
from ibapi.contract import *
from ibapi.ticktype import *
import predicting_stocks.Common as Cm

PATH = "Historical_data/last_price.json"


class TestApp(wrapper.EWrapper, EClient):
    def __init__(self, ticker):
        wrapper.EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self.nextValidOrderId = None
        self.count = 0
        self.ticker = ticker

    @iswrapper
    def nextValidId(self, order_id: int):
        print("nextValidOrderId:", order_id)
        self.nextValidOrderId = order_id

        # here is where you start using api
        contract = Contract()
        contract.symbol = self.ticker
        contract.secType = "STK"
        contract.currency = "USD"
        contract.exchange = "SMART"
        self.reqMarketDataType(3)
        self.reqMktData(1101, contract, "", False, False, [])

    @iswrapper
    def error(self, req_id: TickerId, error_code: int, error_string: str):
        print("Error. Id: ", req_id, " Code: ", error_code, " Msg: ", error_string)

    @iswrapper
    def tickPrice(self, req_id: TickerId, tick_type: TickType, price: float,
                  attrib: TickAttrib):
        # print("Tick Price. Ticker Id:", req_id,
        #       "tickType:", TickTypeEnum.to_str(tick_type),
        #       "Price:", price)
        Cm.write_in_json(PATH, {self.ticker: {"price": price}})
        # just disconnect after a bit
        self.count += 1
        if self.count > 1:
            self.disconnect()


# I use jupyter but here is where you use if __name__ == __main__:
def get_last_price(ticker):
    app = TestApp(ticker)
    app.connect("127.0.0.1", 4002, clientId=123)
    # print(app.serverVersion(), app.twsConnectionTime())
    app.run()
    return Cm.get_from_interactive(path=PATH, ticker=ticker, key='price', app=app)


if __name__ == '__main__':
    get_last_price('NIO')

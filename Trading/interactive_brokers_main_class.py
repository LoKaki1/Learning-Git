from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *

import threading
import time


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def nextValidId(self, orderId: int):
        super().nextValidId(orderId)
        self.nextorderId = orderId
        print('The next valid order id is: ', self.nextorderId)

    def orderStatus(self, orderId, status, filled, remaining, avgFullPrice, permId, parentId, lastFillPrice, clientId,
                    whyHeld, mktCapPrice):
        print('orderStatus - orderid:', orderId, 'status:', status, 'filled', filled, 'remaining', remaining,
              'lastFillPrice', lastFillPrice)

    def openOrder(self, orderId, contract, order, orderState):
        print('openOrder id:', orderId, contract.symbol, contract.secType, '@', contract.exchange, ':', order.action,
              order.orderType, order.totalQuantity, orderState.status)

    def execDetails(self, reqId, contract, execution):
        print('Order Executed: ', reqId, contract.symbol, contract.secType, contract.currency, execution.execId,
              execution.orderId, execution.shares, execution.lastLiquidity)


def run_loop():
    app.run()


# Function to create FX Order contract
def FX_order(symbol):
    contract = Contract()
    contract.symbol = symbol
    contract.secType = 'STK'
    contract.exchange = 'SMART'
    contract.currency = 'USD'
    return contract


app = IBapi()
app.connect('127.0.0.1', 7497, 123)

app.nextorderId = None

# Start the socket in a thread
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

# Check if the API is connected via orderid
while True:
    if isinstance(app.nextorderId, int):
        print('connected')
        break
    else:
        print('waiting for connection')
        time.sleep(1)


# Create order object
def create_order(action='BUY', quantity=100, order_type='LMT', limit_price=0):
    order = Order()
    order.action = action
    order.totalQuantity = quantity
    order.orderType = order_type
    order.lmtPrice = limit_price
    return order


def place_order(ticker, ap=app, action='BUY', quantity=100, order_type='LMT', limit_price=0):
    # Place order
    ap.placeOrder(app.nextorderId, FX_order(ticker), create_order(action, quantity, order_type, limit_price))
    app.nextorderId += 1


time.sleep(3)
place_order('NIO', limit_price=30)
# Cancel order
print('cancelling order')
app.cancelOrder(app.nextorderId)

time.sleep(3)
app.disconnect()
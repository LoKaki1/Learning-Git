from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import time
import threading

class MyWrapper(EWrapper):
    def __init__(self):
        super().__init__()
        self.data = []
        self.df = None

    def nextValidId(self, orderId: int):
        print("Setting nextValidOrderId: %d", orderId)
        self.nextValidOrderId = orderId
        self.start()

    def historicalData(self, reqId, bar):
        self.data.append(vars(bar));

    def historicalDataUpdate(self, reqId, bar):
        line = vars(bar)
        # pop date and make it the index, add rest to df
        # will overwrite last bar at that same time
        self.df.loc[pd.to_datetime(line.pop('date'))] = line

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        print("HistoricalDataEnd. ReqId:", reqId, "from", start, "to", end)
        self.df = pd.DataFrame(self.data)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.set_index('date', inplace=True)

    def error(self, reqId, errorCode, errorString):
        print("Error. Id: ", reqId, " Code: ", errorCode, " Msg: ", errorString)

    def start(self):
        queryTime = ""

        # so everyone can get data use fx
        fx = Contract()
        fx.secType = "STK"
        fx.symbol = "NIO"
        fx.currency = "USD"
        fx.exchange = "SMART"

        # setting update to 1 minute still sends an update every tick? but timestamps are 1 min
        # I don't think keepUpToDate sends a realtimeBar every 5 secs, just updates the last bar.
        app.reqHistoricalData(1, fx, queryTime, "1 D", "1 min", "MIDPOINT", 0, 1, True, [])


wrap = MyWrapper()
app = EClient(wrap)
app.connect("127.0.0.1", 7497, clientId=999)

# I just use this in jupyter so I can interact with df


threading.Thread(target=app.run).start()

# this isn't needed in jupyter, just run another cell

# time.sleep(900)  # in 5 minutes check the df and close
timing = time.time() + 2900
while timing > time.time():
    try:
        wrap.df.to_csv("myfile4.csv")  # save in file
        print(wrap.df)
        break
    except AttributeError:
        print('still waiting (please learn async maybe it is related to that)', wrap.df)

    time.sleep(20)

app.disconnect()

# in jupyter to show plot
# % matplotlib
# inline
wrap.df.close.plot()
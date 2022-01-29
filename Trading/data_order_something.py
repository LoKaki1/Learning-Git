from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract
import datetime as dt
import pandas as pd
import time
import threading
import os


class MyWrapper(EWrapper, EClient):
    def __init__(self, ticker, other='3'):
        super().__init__()
        self.nextValidOrderId = None
        self.data = []
        self.df = None
        self.ticker = ticker
        self.wrapper = EWrapper()
        self.app = EClient(self)
        self.other = other
        self.app.connect("127.0.0.1", 7497, clientId=999)

    def nextValidId(self, order_id: int):
        print("Setting nextValidOrderId: %d", order_id)
        self.nextValidOrderId = order_id
        self.start()

    def historicalData(self, req_id, bar):
        self.data.append(vars(bar))

    def historicalDataUpdate(self, req_id, bar):
        line = vars(bar)
        # pop date and make it the index, add rest to df
        # will overwrite last bar at that same time
        self.df.loc[pd.to_datetime(line.pop('date'))] = line

    def historicalDataEnd(self, req_id: int, start_date: str, end_date: str):
        print("HistoricalDataEnd. ReqId:", req_id, "from", start_date, "to", end_date)
        self.df = pd.DataFrame(self.data)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.set_index('date', inplace=True)

    def error(self, req_id, error_code, error_string):
        print("Error. Id: ", req_id, " Code: ", error_code, " Msg: ", error_string)

    def start(self,):
        query_time = ""
        # so everyone can get data use fx
        fx = Contract()
        fx.secType = "STK"
        fx.symbol = self.ticker
        fx.currency = "USD"
        fx.exchange = "SMART"

        # setting update to 1 minute still sends an update every tick? but timestamps are 1 min
        # I don't think keepUpToDate sends a realtimeBar every 5 secs, just updates the last bar.
        last_date = get_last_and_first_dates(self.ticker)[-1]
        if last_date == '':
            print('fuck no data on this ticker')
            self.app.reqHistoricalData(1, fx, query_time, f"{self.other} D", "1 min", "MIDPOINT", 0, 1,
                                       True, [])
            return
        now = dt.datetime.now()
        # 2022-01-21 11:00:00
        now_less_last_seconds = (now - dt.datetime.strptime(last_date, '%Y-%m-%d %H:%M:%S')).total_seconds()
        print(now_less_last_seconds)
        self.app.reqHistoricalData(1, fx, query_time, f"{int(now_less_last_seconds)} S", "1 min", "MIDPOINT", 0, 1, True, [])


def get_last_and_first_dates(ticker):
    """
    Return:  The last and first values o
    """
    return [(p := pd.read_csv(path)['date'])[0], p[len(p) - 1]]\
        if os.path.exists((path :=
            f'../Trading/Historical_data/{ticker}.csv')) else ['']


def read_from_file(ticker):
    try:
        with open(f'../Trading/Historical_data/{ticker}.csv', 'r') as file:
            data = file.read()
            file.close()
            return data
    except FileNotFoundError:
        print('file not found')
        return None


def read_data(ticker='NIO', other='3'):
    app = MyWrapper(ticker, other).app

    threading.Thread(target=app.run).start()
    timing = time.time() + 2900
    path = None
    while timing > time.time():
        try:
            app.wrapper.df.to_csv((path := f"../Trading/Historical_data/{ticker}.csv"), mode='a')  # save in file
            print(app.wrapper.df)
            break
        except AttributeError:
            print('still waiting (please learn async maybe it is related to that)', app.wrapper.df)

            time.sleep(1)

    app.disconnect()
    app.wrapper.df.close.plot()
    return path

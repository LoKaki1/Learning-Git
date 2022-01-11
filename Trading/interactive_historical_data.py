from ibapi import wrapper, client
from ibapi.common import BarData
import time
import datetime
import threading
from data_order_something import create_contract

class TestWrapper(wrapper.EWrapper, client.EClient):

    def __init__(self):

        wrapper.EWrapper.__init__(self)
        client.EClient.__init__(self, self)

    def historicalData(self, reqId:int, bar: BarData):
        print("HistoricalData. ReqId:", reqId, "BarData.", bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        super().historicalDataEnd(reqId, start, end)
        print("HistoricalDataEnd. ReqId:", reqId, "from", start, "to", end)

    def historicalDataUpdate(self, reqId: int, bar: BarData):
        print("HistoricalDataUpdate. ReqId:", reqId, "BarData.", bar)


def run_loop():
    app.run()

app = TestWrapper()
app.connect('127.0.0.1', 7497, 123)

app.next_order_id = None

# Start the socket in a thread
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

# Check if the API is connected via orderid
while True:
    if isinstance(app.next_order_id, int):
        print('connected')
        break
    else:
        print('waiting for connection')
        time.sleep(1)
end = (datetime.datetime.today()).strftime('%Y%m%d  %H:%M:%S')
print(app.reqHistoricalData(4001, create_contract('NIO'), endDateTime=end, durationStr='6 M',
                            barSizeSetting='1 min', whatToShow='MIDPOINT', useRTH=1, formatDate=1, keepUpToDate=True,
                            chartOptions=[]))
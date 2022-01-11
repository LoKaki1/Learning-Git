import asyncio

from ib_insync import *

timeout = 10
ib = IB()
ib.connect()
contract = Stock('NIO', exchange='SMART', currency='USD')

bars = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='6 M',
    barSizeSetting='1 min', whatToShow='MIDPOINT', useRTH=True, timeout=30
)

data_frame = util.df(bars)
print(data_frame)
from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.scanner import ScannerSubscription
from ibapi.tag_value import TagValue
import time
import threading
from Trading import premarket_data
from predicting_stocks import Common as Cm


PATH = 'scanner.json'
class ScannerReader(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.counter = 0

    def scannerData(self, req_id: int, rank: int, contract_details,
                    distance: str, benchmark: str, projection: str, legs_str: str):
        # super().scannerData(req_id, rank, contract_details, distance, benchmark, projection, legs_str)

        # print(f"Scanner data request: {req_id}, {contract_details}, {rank}")

        ticker = contract_details.contract.symbol
        last_price = Cm.get_last_price(ticker)
        premarket_price = premarket_data.get_last_price(ticker)
        change = (premarket_price - last_price) / last_price * 100
        Cm.write_in_json(PATH, {'status': 'done'})
        Cm.write_in_json(PATH, {
            f'{ticker}':
            {
                "price": premarket_price,
                'Change %': change
            }
        }
                         )
        self.counter += 1


def stock_scanner(asset_type='STK', asset_loc='STK.NASDAQ', scan_code='TOP_PERC_GAIN'):
    scan_sub = ScannerSubscription()
    scan_sub.numberOfRows = 50
    scan_sub.belowPrice = 20
    scan_sub.aboveVolume = 500000

    scan_sub.instrument, scan_sub.locationCode, scan_sub.scanCode = asset_type, asset_loc, scan_code
    return scan_sub


def start(app):
    app.run()


def get_most_gain_scanner(**kwargs):

    scanner = 'changePercAbove' if 'scanner' not in kwargs else kwargs['scanner']
    scanner_args = '15' if 'scanner_args' not in kwargs else kwargs['scanner_args']
    Cm.write_in_json(PATH, {'status': 'not done'})
    app = ScannerReader()
    app.connect('localhost', 4002, 222)
    thread = threading.Thread(target=start, args=[app])
    thread.start()
    tag_values = TagValue(scanner, scanner_args)
    app.reqScannerSubscription(3, stock_scanner(), [], scannerSubscriptionFilterOptions=[tag_values])
    for i in range(100):
        if i > 10 and Cm.open_json(PATH)['status'] == 'done':
            break
        time.sleep(1)

    return Cm.get_from_interactive(path=PATH, app=app)


if __name__ == '__main__':
    print(get_most_gain_scanner())

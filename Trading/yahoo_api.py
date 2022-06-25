import time
from selenium import webdriver, common
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# import predicting_stocks.Common as Cm

CHROME_DRIVER_PATH = 'D:/chromedriver.exe'


def _generate_chrome_driver(background_process):

    mini_option = webdriver.ChromeOptions()
    mini_option.add_argument('--headless')

    return webdriver.Chrome(
        CHROME_DRIVER_PATH, options=mini_option) if background_process else\
        webdriver.Chrome(
        CHROME_DRIVER_PATH,)


class LiveData:
    def __init__(self, ticker, background_process=True):
        self.chrome = _generate_chrome_driver(background_process)
        self.chrome.get(f'https://finance.yahoo.com/quote/{ticker}?p={ticker}')
        self.price_xpath = '//*[@id="quote-header-info"]/div[3]/div[1]/div[1]/fin-streamer[1]'
        self.pre_price_xpath = '//*[@id="quote-header-info"]/div[3]/div[1]/div[2]/fin-streamer[2]'

    def get_live_price(self):
        try:
            price = self.chrome.find_element(by='xpath', value=self.pre_price_xpath)
        except common.exceptions.NoSuchElementException:
            price = self.chrome.find_element(by='xpath', value=self.price_xpath)
        return float(price.text)

#
# live = LiveData('NIO')
# while True:
#     print(live.get_live_price())
#     time.sleep(2)

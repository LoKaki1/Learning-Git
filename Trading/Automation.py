import os
import whatsapp
import ai_and_stuff
import time

SLEEPING_TIME = 2 * 60


def main_loop():
    while True:
        getting_request()
        time.sleep(SLEEPING_TIME)


def getting_request():
    data = whatsapp.read_email_from_gmail()
    data = data.strip(" ")
    data = data.strip("]</div>")
    print(data)
    if data.startswith('['):
        data = data.strip('[')

        stocks_list = data.split(', ')
        for i in range(len(stocks_list)):
            stocks_list[i] = stocks_list[i].split(']')[0]
            print(stocks_list[i])
            if stocks_list[i].count('&#39;'):
                stocks_list[i] = stocks_list[i].split('&#39;')[1]

        body = ai_and_stuff.predict_stocks(stocks_list)


if __name__ == '__main__':
    main_loop()

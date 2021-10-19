from fucck import predict_stocks
from whatsapp import write_in_file, send_email
import tensorflow


stocks = ['NIO', 'XPEV', 'LI', 'MARA', "RIOT"]

print(tensorflow.__version__)

for  i in stocks:
    pre = predict_stocks([i], '147', prediction_days='80', prediction_day='1')
    stock_data = '\n' + str(i) + 'is gonna be at - ' + str(pre)
    print(stock_data)
    write_in_file(path='stocks.txt', data=stock_data)
    send_email(stock_data)



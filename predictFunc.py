from fucck import predict_stocks
from whatsapp import write_in_file

stocks_list = ['LI']
               #['NIO 'XPEV', 'LI', 'TSLA', 'RIOT', 'MARA']
file_path = 'prediction.txt'
write_in_file('s', file_path)
for i in stocks_list:
    prediction = predict_stocks([i], units='147', prediction_day='1', prediction_days='80')
    data = '\n' + 'The Stock ' + i + ' is gonna be today at ' + str(prediction)
    print(data)
    write_in_file(data, file_path)
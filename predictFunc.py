from fucck import predict_stocks, test_end
from Evolution import write_in_file

stocks_list = ['NIO', 'XPEV', 'LI', 'MARA', 'RIOT']
               #['NIO 'XPEV', 'LI', 'TSLA', 'RIOT', 'MARA']

file_path = 'prediction.txt'
write_in_file('s', file_path)
for i in stocks_list:
    prediction = predict_stocks([i], epochs=20, batch_size=110, units=140, prediction_days=80)
    data = '\n' + 'The Stock ' + i + ' is gonna be today at ' + str(prediction) + ' at the date ' + test_end
    print(data)
    write_in_file(data=data, path=file_path)

    # need to learn what is batch size
    # because I said so
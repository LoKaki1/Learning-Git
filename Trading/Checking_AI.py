from sklearn.preprocessing import MinMaxScaler

import ai_and_stuff
import datetime as dt
from getting_data import get_historical_data


def read_from_prediction(path):
    file = open(path, 'r')
    data = file.read()
    file.close()
    data = data.strip('----').strip('\n')
    data = data.split('(')
    real_data = {}
    for i in data:
        try:
            j = i.split(', ')
            # print(j[1].strip("'Predict Price - '"))
            price = float(j[1].strip("'Predict Price - '"))
            real_data[j[0].strip("'").strip("'")] = price
        except Exception as e:
            pass
        finally:
            pass

    return real_data


def ai_checking(what, path='Prediction.txt'):
    prediction_stocks = read_from_prediction(path)
    real_values = []
    real_prediction = {}
    stocks = []
    problem = []
    for i in prediction_stocks.keys():
        stocks.append(i)
    for i in stocks:
        if len(i) <= 4:
            x = get_historical_data(i.strip("'"))[what].values[-1]
            real_values.append(x)
            real_prediction[i.strip("'")] = x
        else:
            problem.append(i)

    for i in problem:
        stocks.remove(i)

    co = 0
    
    for i in range(min(len(stocks), len(real_values)) - 1):
        co += min(prediction_stocks[stocks[i]]/real_values[i], real_values[i]/prediction_stocks[stocks[i]])
    al = {}
    c = 0
    for i in stocks:

        al[i] = (real_values[c], prediction_stocks[i])
        c += 1
    print(al)
    return co/min(len(stocks), len(real_values)),



def main():
    print(read_from_prediction('Prediction.txt'))
    print(ai_checking('close'))

if __name__ == '__main__':
    main()
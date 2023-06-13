import joblib
import numpy as np
from flask import Flask, request, render_template


# перевод входных данных (введенных пользователем) из строки в массив вещественных чисел
def prepare_data(data_inp):
    x = []
    data = data_inp.split(',')
    for item in data:
        n = float(item)
        x.append(n)
    X_test = np.array(x).reshape(1, -1)
    return(X_test)


# получение предсказаний модели
def get_y_pred(X_test):
    with open('reports/best_model.txt', 'r') as f:
        best_model = str(f.readline())
    loaded_model = joblib.load('reports/' + str(best_model) + '/model.sav')
    y_pred = loaded_model.predict(X_test)
    print(y_pred)
    return (y_pred)


# Функция для перевода id класса в слово
def convert_to_word(value):
    match value:
        case 1  : return("Пшеница")
        case 2  : return("Горчица")
        case 3  : return("Чечевица")
        case 4  : return("Без урожая")
        case 5  : return("Зеленый горошек")
        case 6  : return("Сахарный тростник")
        case 8  : return("Чеснок")
        case 9  : return("Кукуруза")
        case 13 : return("Грамм")
        case 14 : return("Кориандр")
        case 15 : return("Картофель")
        case 16 : return("Берсем")
        case 36 : return("Рис")


app = Flask(__name__)


@app.route('/crop-types', methods=['POST'])
def process_deploy():
    inp = prepare_data(request.data.decode("utf-8"))
    pred = get_y_pred(inp)
    return convert_to_word(pred)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port = 5000)

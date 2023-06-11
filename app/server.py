# написать скрипт, позволяющий: 
# 1 - заходить в репозиторий твой открытый, открывать таблицу с тестовым набором данных, 
# 2 - обрабатывать таблицу (записать в какой-нибцдь тип данных),
# 3 - класть эти данные в модель для прогонки,
# 4 - выдавать результат - таблицу новую (поле-класс), в которой первый столбец - номер поля, второй столбец класс сельхозугодья

import os
import joblib
from flask import Flask, render_template, request
import pandas as pd
import requests
from io import StringIO


# Функция для перевода значения в слово
def convert_to_word(value):
    return {
        1: "Пшеница",
        2: "Горчица",
        3: "Чечевица",
        4: "Без урожая",
        5: "Зеленый горошек",
        6: "Сахарный тростник",
        8: "Чеснок",
        9: "Кукуруза",
        13: "Грамм",
        14: "Кориандр",
        15: "Картофель",
        16: "Берсем",
        36: "Рис"
    }.get(value, "Неизвестное значение")

def load_y_pred():
    with open('../reports/best_model_dir.txt', 'r') as f:
        best_model_dir = str(f.readline())
    model = joblib.load('../reports/' + str(best_model_dir) + '/model.sav')
    X_test = [[43.08695652173913,39.05797101449275,38.42028985507246,38.405797101449274,42.507246376811594,59.53623188405797,68.79710144927536,64.40579710144928,76.1304347826087,12.507246376811594,72.15942028985508,48.768115942028984],
              [46.0,43.95238095238095,45.333333333333336,49.80952380952381,52.42857142857143,65.9047619047619,75.85714285714286,71.28571428571429,84.28571428571429,12.047619047619047,90.66666666666667,65.28571428571429],
              [45.0,42.8125,43.5625,48.0,50.5,61.125,69.8125,64.9375,77.0,12.0,85.125,61.25]] # Пример входных данных
    y_pred = model.predict(X_test) # Получение предсказаний модели
    print(y_pred) # Вывод предсказаний
    stroka = 'Слова:'
    for value in y_pred:
        word = convert_to_word(value)
        print(word)
        stroka = stroka + ' ' + word
    return (stroka)

app = Flask(__name__)

@app.route('/update', methods=['POST'])
def update():
    global y_pred
    y_pred = load_y_pred()
    return(y_pred)
@app.route('/')
def page():
    #update()
    return render_template('index.html')
@app.route('/', methods=['POST'])
def load_table():
    try:
        y_pred = update()
        # # Получаем ссылку на таблицу из запроса
        # table_url = request.form.get('https://github.com/smlrdk/classification-crop-types/blob/main/prepared_test_data.csv')
        # # Отправляем запрос на получение содержимого таблицы
        # response = requests.get(table_url)
        # # Если запрос не был успешным, возвращаем ошибку  Мы д
        # if response.status_code != 200:
        #     return 'Error: Unable to load table'
        # # Получаем содержимое таблицы в виде строки
        # table_str = response.text
        # # Создаем объект pd.DataFrame из строки
        # table_df = pd.read_csv(StringIO(table_str))
        # update()
        return y_pred
    except:
        return "err"
    

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 8080))
#     app.run(debug=True, host='192.168.1.68', port=port)

if __name__ == '__main__':
    # port = int(os.environ.get('PORT', 5000))
    # app.run(debug=True, host='127.0.0.1', port=port)

    app.run(debug=True, host='0.0.0.0')
import os
import sys
import joblib
import pickle
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def get_version(name):
    with open("reports/version_" + str(name) + ".txt") as version_file:
        version = int(version_file.readline())
    return version


def update_version(name):
    with open("reports/version_" + str(name) + ".txt") as version_file:
            version = int(version_file.readline()) + 1
    with open("reports/version_" + str(name) + ".txt", "w") as file:
        print(version, file=file)
    # создание директории для данных о модели
    os.mkdir("reports/" + str(name) + "_version_" + str(version))
    return version


def train(name, model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    v = update_version(name)
    filename = 'reports/' + str(name) + '_version_' + str(v) + '/model.sav'
    joblib.dump(model, open(filename, 'wb'))
    # сохранение данных
    df = pd.DataFrame(y_pred).to_csv('reports/' + str(name) + '_version_' + str(v) + '/prediction.csv', index=False, header=False)


def name_model(name):
    if not os.path.exists('reports'):
        os.makedirs('reports')
    with open('reports/vars_for_train.pickle', 'rb') as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    with open('reports/vars_for_fine_tuning.pickle', 'rb') as f:
        X_retrain, X_retest, y_retrain, y_retest = pickle.load(f)
    v = get_version(name)
    if not os.path.exists('reports/' + str(name) + '_version_' + str(v) + '/model.sav') or v % 2 == 0:
        # настройка соответствующих моделей
        if name == 'KNN':
            model = KNeighborsClassifier(n_neighbors=13)
        elif name == 'RFC':
            # настройка гиперпараметров для количества деревьев
            n_trees = 30
            model = RandomForestClassifier(n_estimators = n_trees, random_state = 45, n_jobs = 15)
        train(name, model, X_train, y_train, X_test)
    else:
        if name == 'KNN':
            model = KNeighborsClassifier(n_neighbors=13)
        model = joblib.load('reports/' + str(name) + '_version_' + str(v) + '/model.sav')
        train(name, model, X_retrain, y_retrain, X_retest)


if __name__ == "__main__":
    name_model(sys.argv[1])

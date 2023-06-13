import sys
import json
import joblib
import pickle
import os.path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay


# получение номеров версий моделей после обучения и дообучения соответственно
def get_version(name):
    with open("reports/version_" + str(name) + ".txt") as version_file:
        version_train_model = int(version_file.readline()) - 1
    with open("reports/version_" + str(name) + ".txt") as version_file:
        version_retrain_model = int(version_file.readline()) 
    return version_train_model, version_retrain_model


# получение подготовленных тестовых данных для оценки из файла "prepared_test_data", 
# который мы получили на этапе dataset_processing
def get_prepared_test_data():
    data = pd.read_csv('reports/prepared_test_data.csv')
    data.to_string()
    return data


# получение обученной и дообученной моделей соответственно
def get_model(version, name):
    filename = 'reports/' + str(name) + '_version_' + str(version) + '/model.sav'
    loaded_model = joblib.load(filename)
    return loaded_model


# получение предсказаний после прогона тестовых данных через модели
def get_y_pred(version, name):
    df = pd.read_csv('reports/' + str(name) + '_version_' + str(version) + '/prediction.csv').transpose().to_numpy()
    res = df.ravel()
    return res


# подсчет метрик, внесение их в таблицу
def metrics_df(name, y_true, y_pred, version, i):
    parameters = ['Model', 'Accuracy', 'Macro avg Precision', 'Macro avg recall', 'Macro avg f1-score', 
                                'Weighted avg Precision', 'Weighted avg recall', 'Weighted avg f1-score']
    if os.path.isfile('reports/metrics.xlsx'):
        df = pd.read_excel('reports/metrics.xlsx')
    else:
        df = pd.DataFrame(columns=['Model', 'Accuracy', 'Macro avg Precision', 'Macro avg recall', 'Macro avg f1-score', 
                                'Weighted avg Precision', 'Weighted avg recall', 'Weighted avg f1-score'])
    y_true = y_true.drop(index = y_true.index[i])
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average = 'macro')
    precision_weighted = precision_score(y_true, y_pred, average = 'weighted')
    recall_macro = recall_score(y_true, y_pred, average = 'macro')
    recall_weighted = recall_score(y_true, y_pred, average = 'weighted')
    fscore_macro = f1_score(y_true, y_pred, average = 'macro')
    fscore_weighted = f1_score(y_true, y_pred, average = 'weighted')
    model_version = str(name) + "_version_" + str(version)
    new_row = pd.DataFrame([model_version, accuracy, precision_macro, recall_macro, fscore_macro, precision_weighted, 
                            recall_weighted, fscore_weighted], index=parameters).T
    df = pd.concat((df, new_row))
    df.to_excel('reports/metrics.xlsx', index=False)


# получение вероятностей принадлежности полей к каждому классу сельхозугодья
def get_submission(name, model, test_data_grouped, version):
    # извлечение словаря лейблов crop_id
    with open('ref_agrifieldnet_competition_v1/ref_agrifieldnet_competition_v1_labels_train/ref_agrifieldnet_competition_v1_labels_train_001c1/ref_agrifieldnet_competition_v1_labels_train_001c1.json') as ll:
        label_json = json.load(ll)
    # формирование словаря с номером класса и его названием
    crop_dict = {asset.get('value'):asset.get('name') for asset in label_json['assets']['raster_labels']['classification:classes']}
    predictions = model.predict_proba(test_data_grouped.drop('field_id', axis=1))
    crop_columns = [crop_dict.get(i) for i in model.classes_]
    test_df = pd.DataFrame(columns = ['field_id'] + crop_columns)
    test_df['field_id'] = test_data_grouped.field_id
    test_df[crop_columns] = predictions 
    test_df.to_csv('reports/' + str(name) + '_version_' + str(version) + '/submission.csv', index=False)


# подсчет метрик
def evaluation(name_model):
    # извлечение переменных с данными по выборкам (обучающей, дообучающей)
    with open('reports/vars_for_train.pickle', 'rb') as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    with open('reports/vars_for_fine_tuning.pickle', 'rb') as f:
        X_retrain, X_retest, y_retrain, y_retest = pickle.load(f)
    # получение версий моделей (после обучения и дообучения соответственно)
    version_train, version_retrain = get_version(name_model)
    # получение тестовых данных из файла, который мы образовали на этапе dataset_processing
    test_data = get_prepared_test_data()
    # получение обученной и доообученной моделей соответственно
    model_train = get_model(version_train, name_model)
    model_retrain = get_model(version_retrain, name_model)
    # предсказаний после прогона тестовых данных через модели
    y_pred_crop = get_y_pred(version_train, name_model)
    y_re_pred_crop = get_y_pred(version_retrain, name_model)
    index_elements1 = 1
    # вычисление вероятностей принадлежности полей к каждому классу сельхозугодья
    get_submission(name_model, model_train, test_data, version_train)
    get_submission(name_model, model_retrain, test_data, version_retrain)
    # подсчет метрик
    metrics_df(name_model, y_test, y_pred_crop, version_train, index_elements1)
    metrics_df(name_model, y_retest, y_re_pred_crop, version_retrain, index_elements1)
    # cm = get_confMatrix(name_model, y_pred_crop, d.y_test, version_train, index_elements1)
    # cm = get_confMatrix(name_model, y_re_pred_crop, d.y_retest, version_retrain, index_elements2)


if __name__ == "__main__":
    evaluation(sys.argv[1])

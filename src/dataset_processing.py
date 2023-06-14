import os
import json
import pickle
import shutil
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
import dataset_loading as dl
from sklearn.model_selection import train_test_split


# сопоставление номера поля (участка) с типом культуры
def field_crop_extractor(crop_field_files):
    field_crops = {}
    tmp = crop_field_files[:len(crop_field_files)-2]
    for label_field_file in tqdm(tmp):
          with rasterio.open(f'{dl.main}/{dl.train_label_collection}/{dl.train_label_collection}_{label_field_file}/field_ids.tif') as src:
              field_data = src.read()[0]
          with rasterio.open(f'{dl.main}/{dl.train_label_collection}/{dl.train_label_collection}_{label_field_file}/raster_labels.tif') as src:
              crop_data = src.read()[0]
      
          for x in range(0, crop_data.shape[0]): # ширина
              for y in range(0, crop_data.shape[1]):
                  field_id = str(field_data[x][y])
                  field_crop = crop_data[x][y]

                  if field_crops.get(field_id) is None:
                      field_crops[field_id] = []
                  if field_crop not in field_crops[field_id]:
                      field_crops[field_id].append(field_crop)
    field_crop_map  =[[k, v[0]]  for k, v in field_crops.items() ]
    field_crop = pd.DataFrame(field_crop_map , columns=['field_id','crop_id'])
    return field_crop[field_crop['field_id']!='0']


img_sh = 256
n_selected_bands= len(dl.selected_bands)
n_obs = 1  


# создание переменной X таким образом, чтобы каждая строка являлась пикселем, 
# а каждый столбец - одним из наблюдаемых каналов (всего их 12), сопоставленных с соответствующим полем.
def feature_extractor(data_ , path):
    X = np.empty((0, n_selected_bands * n_obs))
    X_tile = np.empty((img_sh * img_sh, 0))
    X_arrays = [] 
    field_ids = np.empty((0, 1))
    tmp = data_['unique_folder_id'][:len(data_['unique_folder_id']) - 2]
    for idx, tile_id in tqdm(enumerate(tmp)):
        if(idx != 1165 and idx != 1166):
          field_src =   rasterio.open( data_['field_paths'].values[idx])
          field_array = field_src.read(1)
          field_ids = np.append(field_ids, field_array.flatten())
          bands_src = [rasterio.open(f'{dl.main}/{path}/{path}_{tile_id}/{band}.tif') for band in dl.selected_bands]
          bands_array = [np.expand_dims(band.read(1).flatten(), axis=1) for band in bands_src]
          X_tile = np.hstack(bands_array)
          X_arrays.append(X_tile)
    X = np.concatenate(X_arrays)
    data = pd.DataFrame(X, columns=dl.selected_bands)
    data['field_id'] = field_ids
    return data[data['field_id']!=0]


# загрузка коллекции json для train и test
with open (f'{dl.main}/{dl.train_label_collection}/collection.json') as f:
    train_json = json.load(f)
with open (f'{dl.main}/{dl.test_label_collection}/collection.json') as f:
    test_json = json.load(f)

# получение всех уникальных идентификаторов папок в папке source для тренировочных и тестовых данных  
train_folder_ids = [i['href'].split('_')[-1].split('.')[0] for i in train_json['links'][4:]]
test_folder_ids = [i['href'].split('_')[-1].split('.')[0] for i in test_json['links'][4:]]

# в соответствии с уникальными идентификаторами взятие из папки train пути к файлам fields_ids.tif и raster_labels.tif
train_field_paths = [f'{dl.main}/{dl.train_label_collection}/{dl.train_label_collection}_{i}/field_ids.tif' for i in train_folder_ids]
train_label_paths = [f'{dl.main}/{dl.train_label_collection}/{dl.train_label_collection}_{i}/raster_labels.tif' for i in train_folder_ids]

# аналогичные действия с папкой test, но только пути к файлам fields_ids.tif
test_field_paths = [f'{dl.main}/{dl.test_label_collection}/{dl.test_label_collection}_{i}/field_ids.tif' for i in test_folder_ids]

# создание таблицы с уникальными идентификаторами папок в папке train и путями к этим папкам
initial_train_data = pd.DataFrame(train_folder_ids, columns=['unique_folder_id'])
initial_train_data['field_paths'] = train_field_paths

# аналогичные действия, но только в папке test
competition_test_data = pd.DataFrame(test_folder_ids , columns=['unique_folder_id'])
competition_test_data['field_paths'] = test_field_paths

# вызов функции на сопоставление номера поля (участка) с типом культуры
field_crop_pair = field_crop_extractor(train_folder_ids)

# сопоставление номера поля с значениями пикселей, которые находятся в этом поле для каждого канала 
train_data = feature_extractor(initial_train_data, dl.source_collection)
test_data = feature_extractor(competition_test_data,  dl.source_collection)

# каждое поле имеет несколько пикселей в данных. Цель — построить модель случайного леса (RF), используя средние значения
# пикселей в каждом поле. С помощью `groupby` получаем среднее значение для каждого field_id.
train_data_grouped = train_data.groupby(['field_id']).mean().reset_index()
train_data_grouped.field_id = [str(int(i)) for i in train_data_grouped.field_id.values]

test_data_grouped = test_data.groupby(['field_id']).mean().reset_index()
test_data_grouped.field_id = [str(int(i)) for i in test_data_grouped.field_id.values]

# объединение таблиц train_data и field_crop_pair
train_df = pd.merge(train_data_grouped, field_crop_pair, on='field_id' )

if os.path.exists('ref_agrifieldnet_competition_v1'):
    shutil.copytree('ref_agrifieldnet_competition_v1', 'reports/')

train_df.to_csv('reports/prepared_train_data.csv', index=False)
test_data_grouped.to_csv('reports/prepared_test_data.csv', index=False)

# разделение и сохранение данных для обучения и оценки модели
X_train, X_test, y_train, y_test = train_test_split(train_df.drop(['field_id', 'crop_id'], axis=1), train_df['crop_id'], test_size=0.3, random_state=42)
with open('reports/vars_for_train.pickle', 'wb') as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)

# выделение данных для дообучения модели из обучающей выборки
X_train, X_retrain, y_train, y_retrain = train_test_split(X_train, y_train, train_size = 0.9, random_state=42)
with open('reports/vars_for_fine_tuning.pickle', 'wb') as f:
    pickle.dump((X_train, X_retrain, y_train, y_retrain), f)
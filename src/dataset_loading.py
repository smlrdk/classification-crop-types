import os
from radiant_mlhub import Dataset

# Выбор каналов снимка (всего 12 каналов)
Full_bands = ['B01', 'B02', 'B03', 'B04','B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B11', 'B12']
selected_bands = Full_bands[:]

# прописывание названия датасета, ресурсов, важных путей к коллекциям 
main = 'ref_agrifieldnet_competition_v1' 
assets = ['field_ids','raster_labels']

# пути к разным наборам данных (коллекциям)
source_collection = f'{main}_source'
train_label_collection = f'{main}_labels_train'
test_label_collection = f'{main}_labels_test'

if __name__ == "__main__":

    # Загрузка датасета через MLHUB_API_KEY
    os.environ['MLHUB_API_KEY'] = os.getenv('API_KEY')
    dataset = Dataset.fetch(main) # - интерфейс для скачивания набора данных с названием датасета

    # фильтр по коллекциям
    my_filter = dict(
        ref_agrifieldnet_competition_v1_labels_train=assets,
        ref_agrifieldnet_competition_v1_labels_test=[assets[0]], 
        ref_agrifieldnet_competition_v1_source=selected_bands 
    )

    dataset.download(collection_filter=my_filter)


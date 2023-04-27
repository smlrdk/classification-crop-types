## Проектная работа в рамках дисциплины "Архитектура систем ИИ"

#### Автор: Солодкая Мария (Р4140)

### Классификация типов сельхозкультур на спутниковых снимках

#### Цель - идентификация типов сельскохозяйственных культур в наборе данных спутниковых изображений с несбалансированным классом.

#### Задачи:
1. Проанализировать датасет.
2. Спроектировать архитектуру системы.
3. Подготовить данные для обучения алгоритмов Random Forest Classifier и KNN.
4. Обучить модели на основе обучающей выборки и провести оценку полученных результатов.
5. Подготовить тестовый набор данных и получить оценку прогнозных свойств моделей на новых данных.
6. Выбрать оптимальную модель.
7. Выполнить развертывание наилучшей модели.

#### Характеристика датасета:

Данные разбиты на плитки размером 256x256 пикселей. Общее число плиток - 1217. Поля распределены по всем плиткам, некоторые плитки могут иметь только обучающие или тестовые поля, а некоторые могут иметь и то, и другое. Поскольку метки получены на основе данных, собранных на земле, не все пиксели помечены в каждой плитке. Если идентификатор поля для пикселя установлен в 0, это означает, что пиксель не включен ни в обучающий, ни в тестовый наборы (и, соответственно, метка типа культуры также будет равна 0).

Важно знать, что некоторые поля попадают на несколько плиток (как в обучающих, так и в тестовых наборах), и в этом случае пиксели, связанные с одним и тем же идентификатором поля, будут более чем в одной плитке. 

Набор данных содержит 7081 полей, которые разбиты на обучающий и тестовый наборы (5551 полей в обучающем и 1530 полей в тестовом наборе). (UPD: в ходе исследования было выяснено, что последние несколько полей в обучающем наборе данных повреждены, поэтому там только 5538 полей). 

**Данные структурированы в три коллекции:**
* **исходная коллекция (source):** содержит все исходные изображения как для тренировочных данных, так и для тестовых наборов;
* **коллекция меток обучающей выборки (train):** содержит идентификаторы полей и идентификаторы культур для пикселей, поле которых находится в обучающей выборке; 
* **коллекция тестовых меток (test):** содержит только идентификаторы полей для пикселей, поле которых находится в тестовой коллекции.

**Каждая плитка имеет:**
* изображения Sentinel-2 для 12 каналов [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12] сопоставлены с общей сеткой пространственного разрешения 10 м. - находятся в папке source.
* растровый слой, указывающий идентификатор культуры для полей в обучающем наборе данных. - raster_labels.tif
* растровый слой, указывающий идентификаторы полей для полей (обучающих и тестовых наборов). - field_ids.tif

**В метках идет сопоставление пикселей с метками типов культур. Следующие значения пикселей соответствуют следующим типам культур:**

* 1 - Пшеница
* 2 - Горчица
* 3 - Чечевица
* 4 - Без урожая/Залежь
* 5 - Зеленый горошек
* 6 - Сахарный тростник
* 8 - Чеснок
* 9 - Кукуруза
* 13 - Грамм
* 14 - Кориандр
* 15 - Картофель
* 16 - Берсем
* 36 - Рис

<img src="https://github.com/smlrdk/classification-crop-types/blob/main/img_readme/class_distribution.png" width="400">

Рисунок 1. Распределение классов сельхоз культур в датасете


<img src="https://github.com/smlrdk/classification-crop-types/blob/main/img_readme/feature_correlation.png" width="800">

Рисунок 2. Корреляция признаков


<img src="https://github.com/smlrdk/classification-crop-types/blob/main/img_readme/deployment.png">

Рисунок 3. Deployment diagram


<img src="https://github.com/smlrdk/classification-crop-types/blob/main/img_readme/workflow.png">

Рисунок 4. Workflow diagram

#### Источники:
* [Датасет](https://mlhub.earth/data/ref_agrifieldnet_competition_v1)
* [Репозиторий проекта](https://github.com/smlrdk/classification-crop-types)

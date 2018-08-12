ITIL Incident Classifier (Flask)
=====
POST http://127.0.0.1:5000/predict
----------------------------------

Предсказать аналитику по заголовку письма и инициатору

#### Headers

Content-Type: application/json

#### Body

raw (application/json)

    [
        {"id":"1", "title":"Не проводится ТТН", "user":"Диспетчеры склада"},
        {"id":"2", "title":"Не отображается цена", "user":"Иванов Иван"}
    ]

-----
POST http://127.0.0.1:5000/train
----------------------------------

Обучить классификатор на данных из файла

#### Headers

Content-Type: application/json

#### Body

raw (application/json)

    {"file_data": "itil-tickets.csv"}
    
-----
POST http://127.0.0.1:5000/upload
----------------------------------

Загрузить файл для обучения

#### Headers

Content-Type: application/x-www-form-urlencoded

#### Body

formdata

    file
    
-----
GET http://127.0.0.1:5000/upload
----------------------------------

Получить список загруженных файлов

-----
GET http://127.0.0.1:5000/upload?file=itil-tickets.csv
----------------------------------

Скачать загруженный файл

-----
DELETE http://127.0.0.1:5000/upload
----------------------------------
Удалить загруженные файлы

#### Headers

Content-Type: application/json

#### Body

raw (application/json)

    [
    "itil-tickets.csv"
    ]
        
    



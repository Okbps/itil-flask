ITIL Incident Classifier (Flask)
=====
POST http://127.0.0.1:5000/predict
----------------------------------

Предсказать иницитора, услугу и аналитику по заголовку письма

#### Headers

Content-Type

application/json

#### Body

raw (application/json)

    [
        {"id":"1", "title":"не проводится ТТН", "user":"Диспетчеры склада"},
        {"id":"2", "title":"1СРозница - отображение цены", "user":"Дубовик Вадим"} 
    ]


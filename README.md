Flask
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
    	{"id":1, "title":"сайт белкантон"},
    	{"id":2, "title":"эсчф зуп"}
    ]


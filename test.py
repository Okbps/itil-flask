import service, transformers, settings
from pandas import read_json
from numpy import argmax


def train():
    vec, clf = service.train('data/itil-tickets-7.csv')

    transformers.dump_transformer(vec)
    transformers.dump(clf, settings.CLASSIFIER_PATH)


def predict():
    clf = transformers.load(settings.CLASSIFIER_PATH)
    vec = transformers.load_transformer()

    response_data = list()
    j = '''[
        {"id":1, "title":"не проводится ТТН", "user":"Диспетчеры склада"},
        {"id":2, "title":"1СРозница - отображение цены", "user":"Дубовик Вадим"}  
    ]'''

    df_valid = read_json(j)
    df_valid['title'] = df_valid['title'].apply(service.preprocess)

    features = vec.transform(df_valid)
    prediction = clf.predict_proba(features)

    for j in range(len(df_valid)):
        response_row = {"id": df_valid.at[j, "id"], "prediction": dict()}

        for i in range(len(clf.estimators_)):
            indmax = argmax(prediction[i][j])
            response_row["prediction"][settings.Y_COLS[i]] = {
                clf.estimators_[i].classes_[indmax]: prediction[i][j][indmax]}

        response_data.append(response_row)


    print(response_data)


predict()

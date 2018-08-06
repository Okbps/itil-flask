import service
import os
import transformers
import settings
from numpy import argmax
from pandas import DataFrame
from flask import Flask, request, jsonify


app = Flask(__name__)

clf = transformers.load(settings.CLASSIFIER_PATH)
vec = transformers.load_transformer()


@app.route('/predict', methods=['POST'])
def predict():
    response_data = []

    try:
        df_valid = DataFrame(data=request.json)
    except:
        print(type(request.json), request.json)
        return "fail"

    df_valid['title'] = df_valid['title'].apply(service.preprocess)

    features = vec.transform(df_valid)
    prediction = clf.predict_proba(features)

    for j in range(len(df_valid)):
        response_row = {"id": df_valid.at[j, "id"], "prediction": dict()}

        for i in range(len(clf.estimators_)):
            indmax = argmax(prediction[i][j])
            response_row["prediction"][settings.Y_COLS[i]] = {
                "value": clf.estimators_[i].classes_[indmax],
                "prob": prediction[i][j][indmax]
            }

        response_data.append(response_row)

    return jsonify(response_data)


@app.route('/upload', methods=['GET', 'POST', 'DELETE'])
def upload_file():

    if request.method == 'POST':
        f = request.files['file']
        fname = os.path.join(settings.UPLOAD_FOLDER, f.filename)
        f.save(fname)
        return f.filename

    elif request.method == 'GET':
        return jsonify(os.listdir(settings.UPLOAD_FOLDER))

    elif request.method == 'DELETE':
        j = request.json
        removed = []
        for f in j:
            fname = os.path.join(settings.UPLOAD_FOLDER, f)
            try:
                os.remove(fname)
                removed.append(f)
            except:
                pass
        return jsonify(removed)


@app.route('/train', methods=['POST'])
def train():
    j = request.json
    fname = os.path.join(settings.UPLOAD_FOLDER, j['file_data'])
    vec, clf = service.train(fname)

    transformers.dump_transformer(vec)
    transformers.dump(clf, settings.CLASSIFIER_PATH)

    return 'success'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

import utils
import os
from numpy import argmax
from flask import Flask, request, jsonify
from sklearn.externals import joblib


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'data/'
TRANSFORMER_PATH = 'static/itil-tfidf.pkl'
CLASSIFIER_PATH = 'static/itil-multitarget.pkl'

clf = joblib.load(CLASSIFIER_PATH)
transformer = joblib.load(TRANSFORMER_PATH)
transformer._validate_vocabulary()


@app.route('/predict', methods=['POST'])
def predict():
    response_data = []

    for request_row in request.json:

        s = utils.preprocess(request_row['title'])
        features = transformer.transform([s])
        prediction = clf.predict_proba(features.reshape(1, -1))

        response_row = {"id": request_row['id'], "prediction": dict()}

        for i in range(len(clf.estimators_)):
            indmax = argmax(prediction[i])
            response_row['prediction'][clf.estimators_[i].classes_[indmax]] = round(prediction[i][0][indmax], 5)

        response_data.append(response_row)

    return jsonify(response_data)


@app.route('/upload', methods=['GET', 'POST', 'DELETE'])
def upload_file():

    if request.method == 'POST':
        f = request.files['file']
        fname = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(fname)
        return f.filename

    elif request.method == 'GET':
        return jsonify(os.listdir(app.config['UPLOAD_FOLDER']))

    elif request.method == 'DELETE':
        j = request.json
        removed = []
        for f in j:
            fname = os.path.join(app.config['UPLOAD_FOLDER'], f)
            try:
                os.remove(fname)
                removed.append(f)
            except:
                pass
        return jsonify(removed)


@app.route('/train', methods=['POST'])
def train():
    j = request.json
    fname = os.path.join(app.config['UPLOAD_FOLDER'], j['file_data'])
    tfidf, model = utils.train(fname)

    joblib.dump(tfidf, os.path.join(TRANSFORMER_PATH))
    joblib.dump(model, os.path.join(CLASSIFIER_PATH))

    return 'success'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

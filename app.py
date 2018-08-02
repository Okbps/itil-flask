import utils
import os
from numpy import argmax
from flask import Flask, request, jsonify
from sklearn.externals import joblib


app = Flask(__name__)

clf = joblib.load('static/itil-multitarget.pkl')
transformer = joblib.load('static/itil-tfidf.pkl')
transformer._validate_vocabulary()

app.config['UPLOAD_FOLDER'] = 'data/'


@app.route('/predict', methods=['POST'])
def predict():
    j = request.json
    s = utils.preprocess(j['title'])
    features = transformer.transform([s])
    prediction = clf.predict_proba(features.reshape(1, -1))
    rdata = dict()

    for i in range(len(clf.estimators_)):
        indmax = argmax(prediction[i])
        rdata[clf.estimators_[i].classes_[indmax]] = prediction[i][0][indmax]

    return jsonify({'prediction': rdata})


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

    joblib.dump(tfidf, os.path.join('static/itil-tfidf.pkl'))
    joblib.dump(model, os.path.join('static/itil-multinb.pkl'))

    return 'success'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

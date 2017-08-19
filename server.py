#-*-coding:utf-8 -*-
from flask import Flask, render_template, request, jsonify
import os
import json
from models import models
from analysis import preprocessing

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

port = int(os.getenv('PORT', 8000))
db = models.init_db()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tableau')
def tableau():
    return render_template('tableau.html')

@app.route('/upload_page')
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    hr = {}
    target = os.path.join(APP_ROOT, 'data/raw_data')
    if not os.path.isdir(target):
        os.mkdir(target)
    try:
        for file in request.files.getlist('file'):
            filename = file.filename
            destination = "/".join([target, filename])
            file.save(destination)
            hr.update({'response': 'success'})
    except Exception as e:
        print (e)
        hr.update({'response': u'伺服器內部發生錯誤'})
    return render_template('upload.html', **hr)

@app.route('/api/visitors', methods=['POST'])
def put_visitor():
    user = request.json['name']
    return 'Hello {}!'.format(user)

@app.route('/api/save_form', methods=['POST'])
def save_form():
    hr = {}
    try:
        data = request.get_data()
        json_data=json.loads(data)
        models.save_form(db, json_data)
        hr.update({'response': 'success'})
    except Exception as e:
        print (e)
        hr.update({'response': 'error'})
    return json.dumps(hr)

@app.route('/api/get_error_code', methods=['GET'])
def get_error_code():
    hr = {}
    try:
        data = models.get_db(db)
        hr.update({'response': 'success', 'data': data})
    except Exception as e:
        print (e)
        hr.update({'response': 'error'})
    return json.dumps(hr)


def _some_processing():
    try:
        preprocessing.processing_to_csv()
        # preprocessing.processing_to_db(db, 'errorCodeRawDataTable')
        models.import_error_code_raw_data(db)
        models.import_error_code_csv(db)

    except Exception as e:
        print (e)

if __name__ == '__main__':
    _some_processing()
    app.run(host='0.0.0.0', port=port, debug=True)
    
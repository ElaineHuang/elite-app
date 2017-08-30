#-*-coding:utf-8 -*-
from flask import Flask, render_template, request, jsonify
import os
import json
from models import models
from analysis import preprocessing
from datetime import datetime
from analysis import statistic

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
        file = request.files.get('file')
        machine = request.form.get('machine', None)
        save_time = datetime.now().strftime("%Y%m%d%H%M%S")
        # filename = file.filename #更改原來檔名為新的機台名稱待時間
        filename = machine + '_' + save_time + '.xlsx'
        destination = "/".join([target, filename])
        file.save(destination)
        models.get_and_save_new_raw_files(filename, machine)
        hr.update({'response': 'success',
                    'files' : moedls.get_raw_data_filen_name_list()
                    })
    except Exception as e:
        hr.update({'response': u'伺服器內部發生錯誤'})
    return render_template('upload.html', **hr)


@app.route('/api/save_form', methods=['POST'])
def save_form():
    hr = {}
    try:
        data = request.get_data()
        json_data=json.loads(data)
        models.save_form(db, json_data)
        hr.update({'response': 'success'})
    except Exception as e:
        hr.update({'response': 'error'})
    return json.dumps(hr)

@app.route('/api/get_error_code', methods=['GET'])
def get_error_code():
    hr = {}
    try:
        data = models.get_db(db)
        hr.update({'response': 'success', 'data': data})
    except Exception as e:
        hr.update({'response': 'error'})
    return json.dumps(hr)

@app.route('/api/get_new_record', methods=['GET'])
def get_new_record():
    hr = {}
    try:
        data = models.get_new_record(db)
        hr.update({'response': 'success', 'data': data})
    except Exception as e:
        hr.update({'response': 'error'})
    return json.dumps(hr)

@app.route('/api/get_statistic_data', methods=['GET'])
def get_statistic_data():
    hr = {}
    try:
        data = {}
        error_code_data = statistic.error_code_statistic()

        data.update({
            'ErrorCode':error_code_data,
            'ThreeMin': {},
            'FiveMin': {}
        })
        hr.update({'response': 'success', 'data': data})
    except Exception as e:
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
    
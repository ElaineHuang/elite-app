#-*-coding:utf-8 -*-
from flask import Flask, render_template, request, jsonify
import os
import json
from models import models
from analysis import preprocessing, statistic, LightProcessing
from datetime import datetime

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

port = int(os.getenv('PORT', 8000))
db = models.init_db()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/record')
def record():
    return render_template('record.html')

@app.route('/tableau')
def tableau():
    return render_template('tableau.html')

@app.route('/upload_page')
def upload_page():
    return render_template('upload.html')

@app.route('/statistic')
def statistic_page():
    return render_template('statistic.html')

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
        models.some_processing()
        hr.update({'response': 'success',
                    'files' : models.get_raw_data_filen_name_list(target)
                    })
    except Exception as e:
        hr.update({'response': u'伺服器內部發生錯誤'})
    return render_template('upload.html', **hr)

@app.route('/upload_light', methods=['POST'])
def upload_light():
    hr = {}
    target = os.path.join(APP_ROOT, 'data/light_data')
    if not os.path.isdir(target):
        os.mkdir(target)
    try:
        file = request.files.get('file')
        save_time = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = 'light_' + save_time + '.xlsx'
        destination = "/".join([target, filename])
        file.save(destination)
        models.get_and_save_new_raw_files(filename)
        hr.update({'response_light': 'success',
                    'files_light' : models.get_raw_data_filen_name_list(target)
                    })
    except Exception as e:
        hr.update({'response_light': u'伺服器內部發生錯誤'})
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
        error_code_data = statistic.error_code_statistic()
        light_data = statistic.lighter_statistic()
        data = statistic.merge_chart_time_range(error_code_data, light_data)
        hr.update({'response': 'success', 'data': data})
    except Exception as e:
        print e
        hr.update({'response': 'error'})
    return json.dumps(hr)

@app.route('/api/get_danger_error_code', methods=['GET'])
def get_danger_error_code():
    hr = {}
    try:
        danger_error_code = statistic.find_danger_code()
        hr.update({'response': 'success', 'data': danger_error_code})
    except Exception as e:
        print e
        hr.update({'response': 'error'})
    return json.dumps(hr)

@app.route('/api/get_health_index', methods=['GET'])
def health_index():
    hr = {}
    try:
        health_index_result = statistic.calculate_health_index()
        hr.update({'response': 'success', 'data': health_index_result})
    except Exception as e:
        print e
        hr.update({'response': 'error'})
    return json.dumps(hr)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
from flask import Flask, render_template, request, jsonify
import os
import json
from models import models
from analysis import preprocessing

app = Flask(__name__)


port = int(os.getenv('PORT', 8000))
db = models.init_db()


@app.route('/')
def home():
    return render_template('index.html')

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
    
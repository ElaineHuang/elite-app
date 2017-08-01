from flask import Flask, render_template, request, jsonify
import os
import json
from models import models

app = Flask(__name__)


port = int(os.getenv('PORT', 8000))
db = models.init_db()


@app.route('/')
def home():
    # add_test = TestCollection(number=1, name='test')
    # add_test.save()
    # print ("---------------------",add_test.query.first())

    return render_template('index.html')

@app.route('/api/visitors', methods=['POST'])
def put_visitor():
    user = request.json['name']
    return 'Hello {}!'.format(user)

# @app.route('/api/save_form', methods=['POST'])
# def save_form():
#     hr = {}
#     try:
#         test = 1
#         if test:
#            add_test = db.TestCollection(number=1, name='test')
#            add_test.save()
#         else:
#             data = request.get_data()
#             json_data=json.loads(data)
#         hr.update({'response': 'success'})
#     except Exception as e:
#         print (e)
#         hr.update({'response': 'error'})
#     return json.dumps(hr)

@app.route('/api/get_error_code', methods=['POST'])
def get_error_code():
    hr = {}
    try:
        data = models.get_db(db)
        hr.update({'response': 'success', 'data': data})
    except Exception as e:
        print (e)
        hr.update({'response': 'error'})
    return json.dumps(hr)

    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)

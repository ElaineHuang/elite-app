from flask import Flask, render_template, request, jsonify
#from flask.ext.pymongo import PyMongo
from flask_pymongo import PyMongo
import os
import json


app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'elite_factory'
app.config['MONGO_URI'] = 'mongodb://jimpaul15:066husAKW307@ds127783.mlab.com:27783/elite_factory'

mongo = PyMongo(app)

#port = int(os.getenv('PORT', 3080))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/getdata')
def getdata():
	data = mongo.db.elite_factory
	return render_template('data.html')

@app.route('/inputdata', methods=['POST', 'GET'])
def inputdata():
	ef_data = mongo.db.elite_factory
	ef_data.insert({'name':request.form['name'], department:request.form['department']})
	return render_template('index.html')


@app.route('/happy')
def happy():
	return "hello world"

@app.route('/api/visitors', methods=['POST'])
def put_visitor():
    user = request.json['name']
    return 'Hello {}!'.format(user)

if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=port, debug=True)
    #app.secret_key = 'mysecret'
	app.run(debug=True, port=3400)




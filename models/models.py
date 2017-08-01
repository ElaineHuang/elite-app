from flask import Flask
from pymongo import MongoClient, ASCENDING
import csv
import json
import os
import glob
from bson.json_util import dumps

def init_db():
    # app.config['MONGOALCHEMY_DATABASE'] = 'eb8c4f80-3ec6-428d-a4df-ce2011778d7c'
    # app.config['MONGO_URI'] = 'mongodb://d933c021-d8d3-402b-aad2-bb67e8d40601:iEGGAujhUObOBJyFeakrOt3T3@192.168.100.12:27017/eb8c4f80-3ec6-428d-a4df-ce2011778d7c'
    
    #from nick method
	# uri = "mongodb://d933c021-d8d3-402b-aad2-bb67e8d40601:iEGGAujhUObOBJyFeakrOt3T3@192.168.100.12:27017/eb8c4f80-3ec6-428d-a4df-ce2011778d7c"
	# client = MongoClient(uri)
	# db = client['eb8c4f80-3ec6-428d-a4df-ce2011778d7c']

    #localhost db for test
 	uri = "mongodb://127.0.0.1:27017"
	client = MongoClient(uri)
	db = client['tic100']   

	#no useful
    # app.config['MONGOALCHEMY_DATABASE'] = 'tic100'
    # app.config['MONGO_URI'] = 'mongodb://127.0.0.1:27017'
    # db  = MongoAlchemy(app)  

	return db


def import_error_code_csv(db):


	#table
	count = 0
	db.errorCodeTable.drop()
	collect = db['errorCodeTable']
	#CSV to JSON Conversion
	header = ['Repetition', 'ErrCode', 'StartTime']
	print (os.path.abspath(os.path.dirname(__file__)))
	csv_path = os.path.join(os.path.abspath(os.path.dirname('data')), 'data/')

	for file in glob.glob(csv_path + '*_file.csv' ):
		csvfile = open(file, 'r')
		machine_no =  file.split('_')[1]
		reader = csv.DictReader( csvfile )
		print reader
		row = {}
		for each in reader:
			for h in header:
				row.update({
					h: each[h]
				})
			row.update({
				'_id': count,
				'Machine': machine_no,
			})				

			db.errorCodeTable.insert(row)
			count+=1

	print ("done import errorCode")

def get_db(db):
	out_error_db = list(db.errorCodeTable.find())
	return dumps(out_error_db)



from flask import Flask
from pymongo import MongoClient

import csv
import json
import os
import glob
import openpyxl
import xlrd

from bson.json_util import dumps
from datetime import datetime

def init_db():
	#Did't not why this method can't use in EI-pass however local no problem..
    # app.config['MONGOALCHEMY_DATABASE'] = 'eb8c4f80-3ec6-428d-a4df-ce2011778d7c'
    # app.config['MONGO_URI'] = 'mongodb://d933c021-d8d3-402b-aad2-bb67e8d40601:iEGGAujhUObOBJyFeakrOt3T3@192.168.100.12:27017/eb8c4f80-3ec6-428d-a4df-ce2011778d7c'
	#no useful too
    # app.config['MONGOALCHEMY_DATABASE'] = 'tic100'
    # app.config['MONGO_URI'] = 'mongodb://127.0.0.1:27017'
    # db  = MongoAlchemy(app) 
    
    #push to EI-PASS need to open this (from nick method)
	# uri = "mongodb://d933c021-d8d3-402b-aad2-bb67e8d40601:iEGGAujhUObOBJyFeakrOt3T3@192.168.100.12:27017/eb8c4f80-3ec6-428d-a4df-ce2011778d7c"
	# client = MongoClient(uri)
	# db = client['eb8c4f80-3ec6-428d-a4df-ce2011778d7c']

    #localhost db for test
 	uri = "mongodb://127.0.0.1:27017"
	client = MongoClient(uri)
	db = client['tic100']   
	return db


def import_error_code_csv(db):

	db.errorCodeTable.drop()
	collect = db['errorCodeTable']
	#CSV to JSON Conversion
	csv_path = os.path.join(os.path.abspath(os.path.dirname('data')), 'data/useful_data/')
	total = []
	for file in glob.glob(csv_path + '*_file.csv' ):
		csvfile = open(file, 'r')
		machine_no =  file.split('/')[-1].split('.')[0].split('_')[1]
		reader = csv.DictReader( csvfile )
		json_csv = list(reader)
		for r in json_csv:
			r.update({
				'Machine': machine_no
			})
			total.append(r)
		csvfile.close()
	db.errorCodeTable.insert_many(total)

	print ("done import errorCode")

def import_error_code_raw_data(db):

	db.errorCodeRawDataTable.drop()
	collect = db['errorCodeRawDataTable']
	raw_data_path = os.path.join(os.path.abspath(os.path.dirname('data')), 'data/raw_data/')
	data = []
	for file in glob.glob(raw_data_path + 'EQErrRec_*.xlsx' ):
		fileName = file.split('/')[-1].split('.')[0]
		machine_no =  fileName.split('_')[1]
		workbook = xlrd.open_workbook(filename=os.path.join(raw_data_path, file))
		sheet = workbook.sheet_by_name(fileName)
		keys = [v.value for v in sheet.row(0)]
		for row_number in range(sheet.nrows):
			row_data = {}
			for col_number, cell in enumerate(sheet.row(row_number)):
				row_data[keys[col_number]] = cell.value
			row_data.update({
				'Machine': machine_no
			})
			data.append(row_data)
	db.errorCodeRawDataTable.insert_many(data)
	print ("done import errorCodeRawDataTable")


def get_db(db):
	out_error_db = list(db.errorCodeTable.find())
	return dumps(out_error_db)


def save_form(db, data):
	# db.maintainList.drop()
	collect = db['maintainList']
	data.update({
		'update_time': datetime.now()
	})
	db.maintainList.insert(data)
	print ("save form")


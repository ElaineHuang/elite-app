#-*-coding:utf-8 -*-
from flask import Flask
from pymongo import MongoClient

from analysis import preprocessing, statistic, LightProcessing
import csv
import json
import os
import glob
import openpyxl
import xlrd
import time
from bson.json_util import dumps
from datetime import datetime
import pandas as pd


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
	for file in glob.glob(raw_data_path + '*.xlsx' ):
		fileName = file.split('/')[-1].split('.')[0]
		machine_no =  fileName.split('_')[1]
		workbook = xlrd.open_workbook(filename=os.path.join(raw_data_path, file))
		sheet = workbook.sheet_by_name( 'EQErrRec_'+ fileName.split('_')[0])
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

def get_new_record(db):
	new_record = list(db.actionCodeList.find().sort([('update_time', -1)]))
	if len(new_record) > 5:
		new_record = new_record[:5]
	for r in new_record:
		r['event-date'] = r['event-date'].strftime("%Y/%m/%d %H:%M")
		r['update_time'] = r['update_time'].strftime("%Y/%m/%d %H:%M")
	
	return dumps(new_record)


def save_form(db, data):
	collect = db['actionCodeList']
	data['event-date'] = datetime.strptime(data['event-date'], "%Y/%m/%d %H")
	data.update({
		'update_time': datetime.now()
	})
	db.actionCodeList.insert(data)
	print ("save form")


def get_and_save_new_raw_files(filename, machine=None):
	sheet_name = []
	if machine is not None:
		raw_data_path = os.path.join(os.path.abspath(os.path.dirname('data')), 'data/raw_data/')
		useful_data_path = os.path.join(os.path.abspath(os.path.dirname('data')), 'data/useful_data/')
		files_list = sorted(glob.glob(raw_data_path + machine + "*"))
		useful_files_list = sorted(glob.glob(useful_data_path + machine + "*"))
		sheet_name = ['EQErrRec_' + machine]

	else:
		raw_data_path = os.path.join(os.path.abspath(os.path.dirname('data')), 'data/light_data/')
		useful_data_path = os.path.join(os.path.abspath(os.path.dirname('data')), 'data/useful_light_data/')
		files_list = sorted(glob.glob(raw_data_path + 'light_' + "*"))		
		useful_files_list = sorted(glob.glob(useful_data_path + 'light_' + "*"))
		
	df = []
	try:
		for f in files_list:
			if machine is None:
				workbook = openpyxl.load_workbook(filename=os.path.join(raw_data_path, f), read_only=True)			
				for sheetName in workbook.get_sheet_names():
					sheet = workbook.get_sheet_by_name(sheetName)
					if not sheet.cell(row = 1, column=1).value == None:
						sheet_name.append(sheetName)		
			for sheet in sheet_name:
				data = pd.read_excel(f, sheet)
				df.append(data)
		#df list 轉為dataframe
		df = pd.concat(df)
		#把重複的列drop掉
		df = df.drop_duplicates()
		#取最新的檔案當作更新黨
		fileName = files_list[-1].split('/')[-1].split('.')[0]
		#開啟檔案並寫入
		writer_file = pd.ExcelWriter(raw_data_path + fileName +'.xlsx')			
		df.to_excel(writer_file, sheet, index=False)			
		writer_file.save()			

		#刪除舊的機台資訊xlsx csv擋
		os.remove(files_list[0])
			
		for u_f in useful_files_list:
			os.remove(u_f)				


	except Exception as e:
		print e
		raise e

def get_raw_data_filen_name_list(target):
	# raw_data_path = os.path.join(os.path.abspath(os.path.dirname('data')), 'data/raw_data/')
	files = []	
	files_list = sorted(glob.glob(target+ '/' + "*"))
	for f in files_list:
		fileName = f.split('/')[-1].split('.')[0]
		files.append(fileName)

	return files

def some_processing():
    try:
    	print ">>>>?????"
        preprocessing.processing_to_csv()
        # LightProcessing.calculate_light()
        # preprocessing.processing_to_db(db, 'errorCodeRawDataTable')
        # models.import_error_code_raw_data(db)
        # models.import_error_code_csv(db)

    except Exception as e:
        print (e)

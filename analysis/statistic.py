#-*-coding:utf-8 -*-
import pandas as pd
import os
import glob
import copy
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import math
import numpy as np

def error_code_statistic():
	useful_data_path = os.path.join(os.path.abspath(os.path.dirname('data')), 'data/useful_data/')
	files_list = sorted(glob.glob(useful_data_path + "*.csv"))
	files = [f for f in files_list if os.path.isfile(os.path.join(useful_data_path, f))]
	machine_count = {
		'ErrorCode': {}
	}
	for f in files:
		machine_name = os.path.splitext(f)[0].split('/')[-1].split('_')[0]
		df = pd.read_csv(f)
		df['StartTime'] = pd.to_datetime(df['StartTime'],format='%Y/%m/%d %H:%M:%S')
		total_time = df['StartTime']

		min_month = min(total_time).replace(day=1, hour=1, minute=1, second=1)
		max_month = max(total_time).replace(day=1, hour=1, minute=1, second=1)
		t_time = min_month
		time_list = []
		while t_time <= max_month:
		    time_list.append((t_time,t_time + relativedelta(months=1)))
		    t_time += relativedelta(months=1)

		result = {}
		for t in time_list:
		    for s_time in df['StartTime']:
		        if s_time >= t[0] and s_time < t[1]:
		            if s_time.year not in result:
		                result.setdefault(s_time.year, {})
		            if s_time.month not in result[s_time.year]:
		                result[s_time.year].setdefault(s_time.month, {})
		                result[s_time.year].update({
		                    s_time.month: 1
		                })
		            else:
		                result[s_time.year][s_time.month]+=1
		machine_count['ErrorCode'].update({
			machine_name: result
		})

	return machine_count
def lighter_statistic():
	useful_data_path = os.path.join(os.path.abspath(os.path.dirname('data')), 'data/useful_light_data/')
	files_list = sorted(glob.glob(useful_data_path + "*.csv"))
	files = [f for f in files_list if os.path.isfile(os.path.join(useful_data_path, f))]
	light_count = {
		'ThreeMin': {},
		'FiveMin': {}
	}
	count_second = [180, 300] #分別計算三分鐘和五分鐘三色燈亮燈的次數
	for f in files:
		for count_sec in count_second: 
			machine_name = os.path.splitext(f)[0].split('/')[-1].split('_')[0]
			df = pd.read_csv(f)
			df.columns = df.columns.str.strip()
			df['Logtime'] = pd.to_datetime(df['Logtime'],format='%Y/%m/%d %H:%M:%S')

			total_time = df['Logtime']
			min_month = min(total_time).replace(day=1, hour=1, minute=1, second=1)
			max_month = max(total_time).replace(day=1, hour=1, minute=1, second=1)
			t_time = min_month
			time_list = []
			while t_time <= max_month:
			    time_list.append((t_time,t_time + relativedelta(months=1)))
			    t_time += relativedelta(months=1)

			result = {}
			for t in time_list:
				for s_time, duration in zip(df['Logtime'], df['Duration']):
					if s_time >= t[0] and s_time < t[1]:
						if s_time.year not in result:
							result.setdefault(s_time.year, {})
						if s_time.month not in result[s_time.year]:
							result[s_time.year].setdefault(s_time.month, {})
							result[s_time.year].update({
							    s_time.month: 1
							})
						else:
							if duration >= count_sec:
								result[s_time.year][s_time.month]+=1

			if count_sec == 180:
				light_count['ThreeMin'].update({
					machine_name: result
				})

			elif count_sec == 300:
				light_count['FiveMin'].update({
					machine_name: result
				})				

	return light_count

def merge_chart_time_range(error_code_data, light_data):
	#傳入資料資料格式一樣的表結果 去把時間對齊
	#對齊方式很蠢： 取出時間key 做交集..
	year = []
	month = []
	merged_dict = dict(error_code_data, **light_data)
	result = copy.deepcopy(merged_dict)
	for item in merged_dict:
	    for machine in merged_dict[item]:
	        for y in merged_dict[item][machine]:
	            year.extend(merged_dict[item][machine].keys())
	            for m in merged_dict[item][machine][y]:
	                month.extend(merged_dict[item][machine][y].keys()) 

	for item in merged_dict:
		for machine in merged_dict[item]:
			for y in merged_dict[item][machine]:
				for yy in year:
					if yy not in merged_dict[item][machine].keys():
						result[item][machine].update({yy:{}})
				for m in merged_dict[item][machine][y]:
					for mm in month:
						if mm not in merged_dict[item][machine][y]:
							result[item][machine][y].update({mm:0})
	return result

def find_danger_code():
	useful_data_path = os.path.join(os.path.abspath(os.path.dirname('data')), 'data/useful_data/')
	files_list = sorted(glob.glob(useful_data_path + "*.csv"))
	files = [f for f in files_list if os.path.isfile(os.path.join(useful_data_path, f))]
	danger_count = {}
	danger_code = ['E001001', 'E031016', 'E041205', 'W940100', 'W502001', 'W432000', 'W362000']
	now_time = datetime.now()

	for f in files:
		machine_name = os.path.splitext(f)[0].split('/')[-1].split('_')[0]
		df = pd.read_csv(f)
		df['StartTime'] = pd.to_datetime(df['StartTime'],format='%Y/%m/%d %H:%M:%S')
		total_time = df['StartTime']
		min_time = now_time - relativedelta(months=5)

		result = {}
		if not machine_name in danger_count:
			danger_count.setdefault(machine_name, {})
		for error_code, s_time in zip(df['ErrCode'], df['StartTime']):
		    if s_time >= min_time and s_time < now_time:
		        for d_c in danger_code:
		            if d_c not in danger_count:
						danger_count[machine_name].setdefault(d_c, [])		        	
		            if d_c in error_code:
						stf_time = s_time.strftime("%Y/%m/%d %H:%M:%S")
						danger_count[machine_name][d_c].append(stf_time)		                

	return danger_count

def calculate_health_index():
	useful_data_path = os.path.join(os.path.abspath(os.path.dirname('data')), 'data/train_result_data/')
	files_list = sorted(glob.glob(useful_data_path + "*.csv"))
	files = [f for f in files_list if os.path.isfile(os.path.join(useful_data_path, f))]
	health_index = {}

	for f in files:
		machine_name = os.path.splitext(f)[0].split('/')[-1].split('_')[0]
		if not machine_name in health_index:
			health_index.setdefault(machine_name, 0)
		df = pd.read_csv(f)
		for index in df['now_status_predict']:
			if not np.isnan(index):
				if float(index) > 100:
					index = 100
				health_index[machine_name] = math.floor(float(index))

	return health_index

if __name__ == '__main__':
	# error_code_statistic()
	find_danger_code()
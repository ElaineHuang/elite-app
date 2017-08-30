#-*-coding:utf-8 -*-
import pandas as pd
import os
import glob
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def error_code_statistic():
	useful_data_path = os.path.join(os.path.abspath(os.path.dirname('data')), 'data/useful_data/')
	files_list = sorted(glob.glob(useful_data_path + "*.csv"))
	files = [f for f in files_list if os.path.isfile(os.path.join(useful_data_path, f))]
	machine_count = {}
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
		machine_count.update({
			machine_name: result
		})
	return machine_count
def lighter_statistic():
	pass

if __name__ == '__main__':
	error_code_statistic()
#-*-coding:utf-8 -*-
import pandas as pd
import glob
import os

Machines = ['M023', 'M024', 'M025', 'M026']
def merge_train_data_d():
	data_path = os.path.abspath(os.path.dirname('data'))
	train_data = os.path.join(data_path, 'data/train_data/')
	useful_light_data = os.path.join(data_path, 'data/useful_light_data/')
	useful_data = os.path.join(data_path, 'data/useful_data/')
	if not os.path.isdir(train_data):
		os.mkdir(train_data)
	try:
		for machine in Machines:
			light_file = sorted(glob.glob(useful_light_data + machine + "*"))
			error_code_file = sorted(glob.glob(useful_data + machine + "*"))
			if len(light_file)>0:
				light_file = light_file[0]
			if len(error_code_file)>0:
				error_code_file = error_code_file[0]
			df_1 = pd.read_csv(light_file)
			df_2 = pd.read_csv(error_code_file)
			df_1['Situation'] = -1
			df_1.columns = df_1.columns.str.strip()
			df_1 = df_1.rename(index=str, columns={"Situation": "ErrCode", "Logtime": "StartTime"})
			df_3 = pd.concat([df_2[['ErrCode','StartTime']], df_1[['ErrCode','StartTime']]], axis=0)
			df_3['StartTime'] = pd.to_datetime(df_3['StartTime'],format='%Y/%m/%d %H:%M:%S')
			df_3 = df_3.sort_values('StartTime')
			df_3.to_csv(train_data + machine + '_train.csv')
	except Exception as e:
		raise e
if __name__ == '__main__':
	merge_train_data_d()
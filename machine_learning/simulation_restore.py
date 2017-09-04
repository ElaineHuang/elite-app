#-*-coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import pandas as pd
import os
#含 bias


Machines = ['M023']
# , 'M024', 'M025', 'M026']
def add_layer(inputs, in_size, out_size, n_layer,activation_function = None):
	layer_name = 'layer%s' %n_layer
	with tf.name_scope("layer_name"):
		with tf.name_scope("weights"):
			Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
			tf.summary.histogram(layer_name+'/weights',Weights)	
		with tf.name_scope("biases"):
			biases = tf.Variable(tf.zeros([1, out_size])+ 0.1,name='b')
			tf.summary.histogram(layer_name+'/biases',biases)	
		with tf.name_scope("Wx_plus_b"): 
			Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
			tf.summary.histogram(layer_name+'/outputs',outputs)	
		return outputs


def simulation_restore():
	#df=pd.DataFrame(pd.read_excel('M023shutdownErr_300.xlsx'))
	data_path = os.path.abspath(os.path.dirname('../data'))
	train_data = os.path.join(data_path, 'data/train_data/')
	train_result_data = os.path.join(data_path, 'data/train_result_data/')	
	machine_learning_model = os.path.abspath(os.path.dirname('./'))
	for machine in Machines:
		f = train_data + machine + '_train.csv'
		df=pd.DataFrame(pd.read_csv(f))
		# df_simulation=pd.DataFrame(pd.read_excel('simulationErr.xlsx'))


		df = rename_df(df)
		newdata=df[(df['ErrCode'] < 54)] #把54以下的東西萃取出來
		# print(newdata)
		newdata=newdata.pop('ErrCode')  #將newdata的ErrCode那行全部彈出給新的newdata
		#print(df['ErrCode'][1]) #可印出單個格子的數值

		# print(len(newdata))
		#print(newdata.loc[481]) 可以用index把值取出來

		# 我希望index可以重排 http://sofasofa.24xi.org/forum_main_post.php?postid=1000649
		newdata.index = range(len(newdata))

		# print(newdata)

		input_data = pd.DataFrame()
		#row_input_data= pd.DataFrame()
		#input_data = pd.DataFrame([0]*53)
		temp = pd.DataFrame([0]*53)
		# print(input_data)

		stop=[] #紀錄第幾個index停機
		#將每次停機做隔開
		for i in range(len(newdata)):
			if newdata.loc[i]==-1:   #使用索引找值方式
				stop.append(i)   #抓出哪些index是-1
		# print(stop)
		# print(stop[0])
		# print(len(stop))


		g=0
		select_num=0
		end=0
		#stop會列出所有停機的index
		#產生我們要用的輸入資料
		for m in range(len(stop)):
			if m >= 1:
				end = stop[m-1]+1
			for select_num in range(0,stop[m]-end):
				#newdata.loc[k]
				for k in range(g,g+select_num+1):
					for count in range(1,54): #將編好號碼errorcode進行比對看誰有幾個
						if newdata.loc[k] == count:
							temp.loc[count-1]=temp.loc[count-1]+1  #因為編號是1~53 但是dataframe是0~52
							break
				temp.loc[53] = (select_num+1)/(stop[m]-end) #計算output的機率
				#row_input_data=row_input_data.append(temp[0:53],ignore_index=True)  #為了下面需求把input資料都拉成一列
				input_data=pd.concat([input_data,temp],axis=1,ignore_index=True)  #axis=1往右增加 https://tieba.baidu.com/p/3773675591
				temp = pd.DataFrame([0]*53)
			
			g=stop[m]+1

		#將當前狀態做整理，才能預測目前危險指數
		temp2 = pd.DataFrame([0]*53)
		now_status = pd.DataFrame()
		for select_num in range(0,len(newdata)-(stop[len(stop)-1]+1)):
			for mm in range(stop[len(stop)-1]+1,select_num+1+stop[len(stop)-1]+1):
				for count in range(1,54):
					if newdata.loc[mm] == count:
						temp2.loc[count-1] = temp2.loc[count-1]+1
						break
			now_status = pd.concat([now_status,temp2],axis=1,ignore_index=True)	
			temp2 = pd.DataFrame([0]*53)

		now_status = now_status.T



		# print(input_data)
		output_data = input_data.loc[53]   #將輸出資料獨立出來
		input_data = input_data.drop([53]) #將輸入資料獨立出來
		output_data = output_data.T
		input_data = input_data.T   #一列一列排下去 一列有53個

		#該把input_data所有的特徵做特徵正規化，也把當前狀態做正規化
		# print(input_data)
		input_data_normal = (input_data) / (input_data.max() - input_data.min())
		input_data_normal = input_data_normal.fillna(0)  #有些特徵從來沒有數值所以分母在正規化會變成零就會變成NaN
		now_status_normalize = now_status / (input_data.max() - input_data.min())
		now_status_normalize = now_status_normalize.fillna(0)

		upbound =  input_data.max()    #將上下界存起來後面測試資料要用
		lowbound =  input_data.min()
		# print(input_data_normal)


		#print(row_input_data) #輸入資料獨立出來但是變成一整列 沒用到
		# print(output_data) #輸出資料


		#現在狀態預測
		now_status_in = np.array(now_status_normalize[:])



		#可調參數
		batch_size = len(input_data_normal) #總體樣本  ERR300是69

		train_size = 50 #訓練樣本
		train_begin = 0 #訓練起始點
		test_size = batch_size - train_size #測試樣本

		train_input = np.array(input_data_normal[train_begin:train_size])
		#train_input = np.array(input_data[train_begin:train_size])
		train_output=np.array(output_data[train_begin:train_size]).astype(np.float32)
		train_output=np.reshape(train_output,(train_size-train_begin,1))

		test_input = np.array(input_data_normal[train_begin+train_size:batch_size][:])
		#test_input = np.array(input_data[train_begin+train_size:batch_size][:])
		test_output = np.array(output_data[train_begin+train_size:batch_size]).astype(np.float32)
		test_output = np.reshape(test_output,(test_size,1))
		# print(test_input)


		#--------------------------------------------------
		#    機器學習七步驟
		#--------------------------------------------------


		#步驟1使用 
		with tf.name_scope("inputs"):
			xs = tf.placeholder(tf.float32,[None,53],name='x_input')   #1,None代表給多少數字沒關係 後面代表一次輸入多少 
			ys = tf.placeholder(tf.float32,[None,1],name='y_input')  #69[None,1] shape=(69,)

		#步驟2 創建layer
		input_layer=add_layer(xs, 53, 9, n_layer = 1, activation_function = tf.nn.relu)
		hidden_layer1=add_layer(input_layer, 9, 9, n_layer = 2, activation_function = tf.nn.relu6)
		hidden_layer2=add_layer(hidden_layer1, 9, 9, n_layer = 3, activation_function = tf.nn.softmax)
		output_layer=add_layer(hidden_layer2, 9, 1, n_layer=4, activation_function = None)

		#步驟3 loss function

		with tf.name_scope("loss"):
			loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-output_layer),reduction_indices=[1]))
			tf.summary.scalar('loss',loss)

		#步驟3 cross_entropy function
		#http://ithelp.ithome.com.tw/articles/10187002
		#http://studyai.site/page/3
		"""
		with tf.name_scope("cross_entropy"):
			#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(output_layer), reduction_indices=[1]))
			#cross_entropy = tf.cost.cross_entropy(output_layer,ys,'myloss')
			print(output_layer)
			print(ys)
			cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_layer, labels=ys,  name='myloss'))
			#cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ys, logits=output_layer, name='entropy'),'entropy')
			tf.summary.scalar('cross_entropy',cross_entropy)
		"""
		#步驟4 訓練次數
		with tf.name_scope("train"):
			#train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)  #
			train_step=tf.train.AdamOptimizer(0.1).minimize(loss)  #loss
			#train_step=tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)  #cross_entropy
			#train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)  #cross_entropy

		#步驟5 返回模型
		saver = tf.train.Saver()
		sess = tf.Session()
		saver.restore(sess, machine_learning_model +'/'+ machine + '_model/save.ckpt')




		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter("logss/", sess.graph)

		#步驟5.5數據分類
		"""
		DATA = np.array([], dtype=np.float32)
		for ii in range(0,53):
			inputdata = label1[ii,:-1]
			DATA=np.hstack(inputdata)
			print(DATA)
			
		"""


		#print(np.random.rand(1024,1024))


		#ax = fig.add_subplot(1,1,1)
		#
		#plt.show() #印上去程序會暫停

		#plt.ion() #印完不暫停
		#步驟6 開始訓練
		"""
		for i in range(20000):
			
			sess.run(train_step,feed_dict={xs:train_input,ys:train_output})
			if i % 50 == 0:
				result = sess.run(merged,feed_dict={xs:train_input,ys:train_output})
				writer.add_summary(result,i)

				output_ans=sess.run(output_layer,feed_dict={xs:train_input})
				#print(i,sess.run(cross_entropy,feed_dict={xs:train_input,ys:train_output}))
				see_loss = sess.run(loss,feed_dict={xs:train_input,ys:train_output})
				print(i,see_loss)

				#plt.plot([output_ans])
		"""


		#--------------------------------------------------
		#   產出並印出最後一次訓練的結果
		#--------------------------------------------------
		"""
		fig = plt.figure()
		t5=np.linspace(1,train_size,train_size)
		t6=np.linspace(1,train_size,train_size)
		plt.plot(t5, output_ans,'g',linewidth=2)
		plt.plot(t6, train_output,'r',linewidth=2)
		plt.title("forecast shutdown system(Red:true,G:forecast)")
		plt.show()
		"""
		#print(output_ans)
		#print(train_output)

		#--------------------------------------------------
		#   產出並印出測試資料的結果
		#--------------------------------------------------

		# fig = plt.figure()
		test_predict=sess.run(output_layer,feed_dict={xs:test_input})
		#plt.scatter(test_predict, test_output)
		# print(test_input)

		# 如果test_predict超過1變1 小於-1變-1
		test_predict=np.fmin(1.0, np.fmax(0.0, test_predict))  #http://blog.mwsoft.jp/article/176188532.html
		# print(test_predict)
		# 轉成百分比
		# 要帶入人工權重
		artificial = np.array([0.996812, 0.999894, 0.625135, 0.769584, 0.527601, 0.989905, 
			0.596713, 0.956136, 0.334927, 0.68981, 0.186271, 0.51157, 0.967166, 0.00011, 
			0.27513, 0.630326,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
		arti_weight = artificial * test_input
		arti_sum = np.sum(arti_weight,axis=1)  #axis=1代表同列相加
		arti_sum = np.reshape(arti_sum,(len(arti_sum),1))
		# print(arti_weight)
		# print(test_predict)
		arti_sum=np.fmin(1.0, np.fmax(0.0, arti_sum))
		#test_predict = 50 * test_predict + 50 * artificial
		test_predict = 100 * test_predict + 0 * arti_sum
		test_output = 100 * test_output

		#print(test_output)
		#t1=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
		#t2=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
		# t1=np.linspace(1,test_size,test_size)
		# t2=np.linspace(1,test_size,test_size)
		# plt.plot(t1, test_predict,'g',linewidth=2)
		# plt.plot(t2, test_output,'r',linewidth=2)
		# plt.xlabel('status')
		# plt.ylabel('dangerous percentage(%)')
		# plt.title("forecast shutdown system(Red:true,G:forecast)")
		# plt.show()
		#--------------------------------------------------
		#  產出當前狀態危險指數
		#--------------------------------------------------
		now_status_predict=sess.run(output_layer,feed_dict={xs:now_status_in})
		# print('status')
		# print(now_status_predict)



		#--------------------------------------------------
		#  帶入個別 error 找出最重要的 error
		#--------------------------------------------------
		find_important_error = pd.DataFrame(np.random.randn(53,53))
		for i in range(53):
			for j in range(53):
				if i==j:
					find_important_error[i][j]=1
				else:
					find_important_error[i][j]=0
				

		#print(find_important_error)
		#正規化

		find_important_error = find_important_error/(upbound-lowbound)
		# print('haha')
		# print(upbound)
		find_important_error = find_important_error.fillna(0)
		find_important_error = find_important_error.replace(np.inf,0)

		#轉numpy.array
		find_important_error = np.array(find_important_error)

		#轉tensorflow能吃的出入
		# fig = plt.figure()
		individual_error = sess.run(output_layer,feed_dict={xs:find_important_error})
		individual_error=np.fmin(1.0, np.fmax(0.0, individual_error))  #0-1之間
		individual_error = individual_error * 100
		# print(individual_error)
		#如果不曾看過的error 輸出結果就是零 現在的問題是輸入都是零卻還是有危險值輸出
		# upbound會列出1~53 error code編號在所有當機情況中最多出現的次數

		for i in range(53):
			if upbound.loc[i] == 0:
				individual_error[i]=0
			
		# print(individual_error)
		# t3 = np.linspace(1,53,53)
		# new_ticks = np.linspace(1,53,53)
		# plt.plot(t3, individual_error,'g',linewidth=2)
		# plt.xlim((1, 53))
		# plt.ylim((0, 100))
		# plt.xlabel('error code')
		# plt.ylabel('damage percentage')
		# plt.xticks(new_ticks)
		# plt.show()

		#--------------------------------------------------
		#  產生並印出模擬資料
		#--------------------------------------------------
		"""
		fig = plt.figure()
		simulation_predict=sess.run(output_layer,feed_dict={xs:simulation_in})
		#plt.scatter(test_predict, test_output)
		print(simulation_in)

		# 如果test_predict超過1變1 小於-1變-1
		simulation_predict=np.fmin(1.0, np.fmax(0.0, simulation_predict))  #http://blog.mwsoft.jp/article/176188532.html
		print(simulation_predict)
		# 轉成百分比
		# 要帶入人工權重
		artificial = np.array([0.996812, 0.999894, 0.625135, 0.769584, 0.527601, 0.989905, 
			0.596713, 0.956136, 0.334927, 0.68981, 0.186271, 0.51157, 0.967166, 0.00011, 
			0.27513, 0.630326,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
		arti_weight = artificial * simulation_in
		arti_sum = np.sum(arti_weight,axis=1)  #axis=1代表同列相加
		arti_sum = np.reshape(arti_sum,(len(arti_sum),1))
		print(arti_weight)
		print(simulation_predict)
		arti_sum=np.fmin(1.0, np.fmax(0.0, arti_sum))
		#test_predict = 50 * test_predict + 50 * artificial
		simulation_predict = 100 * simulation_predict + 0 * arti_sum
		simulation_out = 100 * simulation_out

		#print(test_output)
		#t1=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
		#t2=[1 2 3 4 5 6 7 8 9 10 11 12 13 14 15]
		t8=np.linspace(1,simulation_batch_size,simulation_batch_size)
		t9=np.linspace(1,simulation_batch_size,simulation_batch_size)
		plt.plot(t8, simulation_predict,'g',linewidth=2)
		plt.plot(t9, simulation_out,'r',linewidth=2)
		plt.xlabel('status')
		plt.ylabel('dangerous percentage(%)')
		plt.title("forecast shutdown system(Red:true,G:forecast)")
		plt.show()
		"""
		#--------------------------------------------------
		#  將 numpy array 存入 csv 檔
		#--------------------------------------------------

		#c=np.concatenate((individual_error,test_predict,test_output,simulation_predict,simulation_out),axis=1)

		# s1=pd.DataFrame(individual_error,columns=['individual_error'])
		# s2=pd.DataFrame(test_predict,columns=['test_predict'])
		# s3=pd.DataFrame(test_output,columns=['test_output'])
		#s4=pd.DataFrame(simulation_predict,columns=['simulation_predict'])
		#s5=pd.DataFrame(simulation_out,columns=['simulation_out'])
		s6=pd.DataFrame(now_status_predict,columns=['now_status_predict'])

		# s1.to_csv('errcode.csv')

		# s_test = pd.concat([s2,s3],axis=1)
		# s_test.to_csv('test.csv')

		#s_simulation = pd.concat([s4,s5],axis=1)
		#s_simulation.to_csv('simulation.csv')

		s6.to_csv(train_result_data + machine + '_status_predict.csv')

		#s_all = pd.concat([s1,s2,s3,s4,s5],axis=1)
		#s_all.to_csv('generate.csv')
		#np.savetxt("foo1.csv", zip(individual_error,test_predict,test_output,simulation_predict,simulation_out), delimiter=',', fmt='%f')










		"""  小用法
		#for j in range(k[0])


		#print(df.columns)
		#print(df.values)

		#print(df['Situation'].isnull())
		#print(df['Situation'].isnull().value_counts())



	#df['Situation'].fillna(value=0)
		#df['Duration'].replace('NaN','1')

		#print(newdata.head(30)) #列出前三十列
		"""	
def rename_df(df):
	for i in range(len(df)):
		if df.loc[i, 'ErrCode'] == 'E620002L':
			df.loc[i, 'ErrCode'] = 54
		elif df.loc[i, 'ErrCode'] == 'E620002R':
			df.loc[i, 'ErrCode'] = 55
		elif df.loc[i, 'ErrCode'] == 'E620003L':
			df.loc[i, 'ErrCode'] = 56		
		elif df.loc[i, 'ErrCode'] == 'E620003R':
			df.loc[i, 'ErrCode'] = 57
		elif df.loc[i, 'ErrCode'] == 'E551010L':
			df.loc[i, 'ErrCode'] = 58
		elif df.loc[i, 'ErrCode'] == 'E551010R':
			df.loc[i, 'ErrCode'] = 59


	#54~59是要剔除的資料
		elif df.loc[i, 'ErrCode'] == 'E632237L':     #0.996812
			df.loc[i, 'ErrCode'] = 1
		elif df.loc[i, 'ErrCode'] == 'E632233L':     #0.999894
			df.loc[i, 'ErrCode'] = 2
		elif df.loc[i, 'ErrCode'] == 'E632501R':     #0.625135
			df.loc[i, 'ErrCode'] = 3
		elif df.loc[i, 'ErrCode'] == 'E632234L':     #0.769584
			df.loc[i, 'ErrCode'] = 4			
		elif df.loc[i, 'ErrCode'] == 'E551502R':     #0.527601
			df.loc[i, 'ErrCode'] = 5
		elif df.loc[i, 'ErrCode'] == 'E632104L':     #0.989905
			df.loc[i, 'ErrCode'] = 6
		elif df.loc[i, 'ErrCode'] == 'E632434R':     #0.596713
			df.loc[i, 'ErrCode'] = 7		
		elif df.loc[i, 'ErrCode'] == 'E632201L':      #0.956136
			df.loc[i, 'ErrCode'] = 8
		elif df.loc[i, 'ErrCode'] == 'E632102L':      #0.334927
			df.loc[i, 'ErrCode'] = 9 
		elif df.loc[i, 'ErrCode'] == 'E551092L':      #0.68981
			df.loc[i, 'ErrCode'] = 10
		elif df.loc[i, 'ErrCode'] == 'E632307R':      #0.186271
			df.loc[i, 'ErrCode'] = 11
		elif df.loc[i, 'ErrCode'] == 'E530020L':   #0.51157
			df.loc[i, 'ErrCode'] = 12
		elif df.loc[i, 'ErrCode'] == 'E632107L':    #0.967166
			df.loc[i, 'ErrCode'] = 13
		elif df.loc[i, 'ErrCode'] == 'E632108L':    #0.00011
			df.loc[i, 'ErrCode'] = 14
		elif df.loc[i, 'ErrCode'] == 'E610045L':    #0.27513
			df.loc[i, 'ErrCode'] = 15
		elif df.loc[i, 'ErrCode'] == 'E632405R':     #0.630326
			df.loc[i, 'ErrCode'] = 16
																	
		elif df.loc[i, 'ErrCode'] == 'E620005L':
			df.loc[i, 'ErrCode'] = 17
		elif df.loc[i, 'ErrCode'] == 'E620007L':
			df.loc[i, 'ErrCode'] = 18

		elif df.loc[i, 'ErrCode'] == 'E551250L':
			df.loc[i, 'ErrCode'] = 19	
		elif df.loc[i, 'ErrCode'] == 'E610056R':
			df.loc[i, 'ErrCode'] = 20

		elif df.loc[i, 'ErrCode'] == 'E600003R':
			df.loc[i, 'ErrCode'] = 21
		elif df.loc[i, 'ErrCode'] == 'E600003L':
			df.loc[i, 'ErrCode'] = 22

		elif df.loc[i, 'ErrCode'] == 'E632236L':
			df.loc[i, 'ErrCode'] = 23

		elif df.loc[i, 'ErrCode'] == 'E551092R':
			df.loc[i, 'ErrCode'] = 24	

		elif df.loc[i, 'ErrCode'] == 'E632436R':
			df.loc[i, 'ErrCode'] = 25

		elif df.loc[i, 'ErrCode'] == 'E551502L':
			df.loc[i, 'ErrCode'] = 26
		elif df.loc[i, 'ErrCode'] == 'E551506R':
			df.loc[i, 'ErrCode'] = 27
		elif df.loc[i, 'ErrCode'] == 'E632103L':
			df.loc[i, 'ErrCode'] = 28
		elif df.loc[i, 'ErrCode'] == 'E530020R':
			df.loc[i, 'ErrCode'] = 29
		elif df.loc[i, 'ErrCode'] == 'E530030L':
			df.loc[i, 'ErrCode'] = 30	

		elif df.loc[i, 'ErrCode'] == 'E605179R':
			df.loc[i, 'ErrCode'] = 31
		elif df.loc[i, 'ErrCode'] == 'E551506L':
			df.loc[i, 'ErrCode'] = 32
		elif df.loc[i, 'ErrCode'] == 'E632301R':
			df.loc[i, 'ErrCode'] = 33
		elif df.loc[i, 'ErrCode'] == 'E530024R':
			df.loc[i, 'ErrCode'] = 34
		elif df.loc[i, 'ErrCode'] == 'E530025R':
			df.loc[i, 'ErrCode'] = 35			
		elif df.loc[i, 'ErrCode'] == 'E610054L':
			df.loc[i, 'ErrCode'] = 36

		elif df.loc[i, 'ErrCode'] == 'E600102R':
			df.loc[i, 'ErrCode'] = 37
		elif df.loc[i, 'ErrCode'] == 'E632304R':
			df.loc[i, 'ErrCode'] = 38	
		elif df.loc[i, 'ErrCode'] == 'E600102L':
			df.loc[i, 'ErrCode'] = 39

		elif df.loc[i, 'ErrCode'] == 'E632504R':
			df.loc[i, 'ErrCode'] = 40

		elif df.loc[i, 'ErrCode'] == 'E610045R':
			df.loc[i, 'ErrCode'] = 41			

		elif df.loc[i, 'ErrCode'] == 'E632205L':
			df.loc[i, 'ErrCode'] = 42
		elif df.loc[i, 'ErrCode'] == 'E632433R':
			df.loc[i, 'ErrCode'] = 43
		elif df.loc[i, 'ErrCode'] == 'E530041R':
			df.loc[i, 'ErrCode'] = 44
		elif df.loc[i, 'ErrCode'] == 'E530014R':
			df.loc[i, 'ErrCode'] = 45	
		elif df.loc[i, 'ErrCode'] == 'E632507R':
			df.loc[i, 'ErrCode'] = 46
		elif df.loc[i, 'ErrCode'] == 'E530041L':
			df.loc[i, 'ErrCode'] = 47
		elif df.loc[i, 'ErrCode'] == 'E601101R':
			df.loc[i, 'ErrCode'] = 48
		elif df.loc[i, 'ErrCode'] == 'E551001L':
			df.loc[i, 'ErrCode'] = 49
		elif df.loc[i, 'ErrCode'] == 'E551001R':
			df.loc[i, 'ErrCode'] = 50			  

		elif df.loc[i, 'ErrCode'] == 'E632101L':
			df.loc[i, 'ErrCode'] = 51
		elif df.loc[i, 'ErrCode'] == 'E620004L':
			df.loc[i, 'ErrCode'] = 52
		elif df.loc[i, 'ErrCode'] == 'E620007R':
			df.loc[i, 'ErrCode'] = 53	

		elif df.loc[i, 'ErrCode']  == "-1":
			df.loc[i, 'ErrCode'] = -1

		else:
			df.loc[i, 'ErrCode'] = 60


	return df

if __name__ == '__main__':
	simulation_restore()

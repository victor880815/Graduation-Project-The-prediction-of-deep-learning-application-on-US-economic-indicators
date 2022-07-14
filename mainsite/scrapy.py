import time,os
import datetime 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable
from sklearn.model_selection import train_test_split


attn_matrix = 128
def attention_net(self, lstm_output, final_state):
	score = self.V(torch.tanh(self.W1(final_state)+self.W2(lstm_output)))
	attention_weights = F.softmax(score, dim=0)
	context_vector = attention_weights * lstm_output
	context_vector = torch.sum(context_vector, dim=0)
	return context_vector, attention_weights


class RNN(nn.Module):
	def __init__(self):
		super(RNN, self).__init__()

		### New layers:
		self.lstm = nn.LSTM(384 ,256, 1, batch_first = True)

		self.lstm2 = nn.LSTM(256 ,128, 1, batch_first = True)
		self.W1 = nn.Linear(attn_matrix, attn_matrix)
		self.W2 = nn.Linear(attn_matrix, attn_matrix)
		self.V = nn.Linear(attn_matrix, 1)
		self.lstm3 = nn.LSTM(132 ,64, 1, batch_first = True)
		self.linear = nn.Linear(64, 1)
		self.sigmoid = nn.Sigmoid()
	    
	def forward(self,x1, x2, x3, x4, x5, x6, x7, his_data):
		out1, (h1,c1) = self.lstm(x1)
		out2, (h2,c2) = self.lstm(x2)
		out3, (h3,c3) = self.lstm(x3)
		out4, (h4,c4) = self.lstm(x4)
		out5, (h5,c5) = self.lstm(x5)
		out6, (h6,c6) = self.lstm(x6)
		out7, (h7,c7) = self.lstm(x7)

		tc = torch.cat((h1,h2,h3,h4,h5,h6,h7),1)
		out, (h,c) = self.lstm2(tc)
		out = torch.reshape(out,(-1,128))
		h = torch.reshape(h,(-1,128))
		context_vector, attention_weights = attention_net(self, lstm_output=out, final_state=h)
		output = context_vector[None, :]
		output = output[None, :]
		output = torch.cat((output,his_data),dim = -1)
		out, (h,c) = self.lstm3(output)
		result = self.linear(h)
		result = self.sigmoid(result)
		return result


def f_label():
	df_F_unemployment = pd.read_excel(r'/home/victor/MIS_finalproject/美國初領失業救濟金人數.xlsx')
	df_F_unemployment = df_F_unemployment.iloc[::-1]
	df_F_unemployment = df_F_unemployment.reset_index(drop = True)
	df_F_unemployment  = df_F_unemployment[3:95]
	df_F_unemployment = df_F_unemployment.reset_index(drop = True)


	tt_list = []
	t_list = df_F_unemployment['announce']
	for data in t_list:
		data = data.replace('K','000')
		data = data.replace(',','')
		data = np.fromstring(data, dtype=float, sep=' ')
		data = np.reshape(data, (-1, 1))
		data = np.array(data).astype("float32").reshape(1, 1, 1)
		data = torch.tensor(data)
		tt_list.append(data)


	former_list = []
	for i in range(0,len(tt_list)-4):
	    tc = torch.cat((tt_list[i],tt_list[i+1],tt_list[i+2],tt_list[i+3]),-1)
	    former_list.append(tc)


	df_F_unemployment = df_F_unemployment[4:]
	df_F_unemployment = df_F_unemployment.reset_index(drop = True)

	return df_F_unemployment, former_list


def c_label():
	df_F_unemployment = pd.read_excel(r'/home/victor/MIS_finalproject/美國續領失業救濟金人數.xlsx')
	df_F_unemployment = df_F_unemployment.iloc[::-1]
	df_F_unemployment = df_F_unemployment.reset_index(drop = True)
	df_F_unemployment  = df_F_unemployment[3:95]
	df_F_unemployment = df_F_unemployment.reset_index(drop = True)


	tt_list = []
	t_list = df_F_unemployment['announce']
	for data in t_list:
		data = data.replace('K','000')
		data = data.replace(',','')
		data = np.fromstring(data, dtype=float, sep=' ')
		data = np.reshape(data, (-1, 1))
		data = np.array(data).astype("float32").reshape(1, 1, 1)
		data = torch.tensor(data)
		tt_list.append(data)


	former_list = []
	for i in range(0,len(tt_list)-4):
	    tc = torch.cat((tt_list[i],tt_list[i+1],tt_list[i+2],tt_list[i+3]),-1)
	    former_list.append(tc)


	df_F_unemployment = df_F_unemployment[4:]
	df_F_unemployment = df_F_unemployment.reset_index(drop = True)

	return df_F_unemployment, former_list


def CNBC():
	df_CF = pd.read_excel(r"/home/victor/MIS_finalproject/CNBC2(Description).xlsx")
	df_CF_concat=(df_CF.groupby('Date',as_index=False).agg(lambda x:list(set(x))[0] if len(set(x))==1 else list(set(x))))
	df_CF_concat = df_CF_concat[:616]
	df_CF_concat = df_CF_concat.reset_index(drop=True)

	return df_CF_concat


def FORBES():
	df_CF = pd.read_excel(r"/home/victor/MIS_finalproject/FORBES2(Summary).xlsx")
	df_CF_concat=(df_CF.groupby('Date',as_index=False).agg(lambda x:list(set(x))[0] if len(set(x))==1 else list(set(x))))
	df_CF_concat = df_CF_concat[:616]
	df_CF_concat = df_CF_concat.reset_index(drop=True)

	return df_CF_concat


def CF():
	df_C = pd.read_excel(r"/home/victor/MIS_finalproject/CNBC2(Description).xlsx")
	df_F = pd.read_excel(r"/home/victor/MIS_finalproject/FORBES2(Summary).xlsx")
	df_CF = pd.concat([df_C,df_F])
	df_CF = df_CF.sort_values(by = 'Date')
	df_CF = df_CF.reset_index(drop=True)
	df_CF_concat=(df_CF.groupby('Date',as_index=False).agg(lambda x:list(set(x))[0] if len(set(x))==1 else list(set(x))))
	df_CF_concat = df_CF_concat[:616]
	df_CF_concat = df_CF_concat.reset_index(drop=True)

	return df_CF_concat


def CNBC_K(keyword):
	df_CF = pd.read_excel(r"/home/victor/MIS_finalproject/CNBC2(Description).xlsx")
	df_CF['Description'] = df_CF.apply(lambda x : ' ' if str(x['Description']) == 'nan' else x['Description'],  axis = 1)
	df_CF = df_CF.reset_index(drop=True)
	df_CF = df_CF[df_CF['Description'].str.contains(keyword)]
	df_CF = df_CF.reset_index(drop=True)

	df_CF_concat=(df_CF.groupby('Date',as_index=False).agg(lambda x:list(set(x))[0] if len(set(x))==1 else list(set(x))))
	df_CF_concat = df_CF_concat[:616]
	df_CF_concat = df_CF_concat.reset_index(drop=True)

	return df_CF_concat


def FORBES_K(keyword):
	df_CF = pd.read_excel(r"/home/victor/MIS_finalproject/FORBES2(Summary).xlsx")
	df_CF['Description'] = df_CF.apply(lambda x : ' ' if str(x['Description']) == 'nan' else x['Description'],  axis = 1)
	df_CF = df_CF.reset_index(drop=True)
	df_CF = df_CF[df_CF['Description'].str.contains(keyword)]
	df_CF = df_CF.reset_index(drop=True)

	df_CF_concat=(df_CF.groupby('Date',as_index=False).agg(lambda x:list(set(x))[0] if len(set(x))==1 else list(set(x))))
	df_CF_concat = df_CF_concat[:616]
	df_CF_concat = df_CF_concat.reset_index(drop=True)

	return df_CF_concat


def CF_K(keyword):
	df_C = pd.read_excel(r"/home/victor/MIS_finalproject/CNBC2(Description).xlsx")
	df_F = pd.read_excel(r"/home/victor/MIS_finalproject/FORBES2(Summary).xlsx")
	df_CF = pd.concat([df_C,df_F])
	df_CF = df_CF.sort_values(by = 'Date')
	df_CF = df_CF.reset_index(drop=True)
	df_CF['Description'] = df_CF.apply(lambda x : ' ' if str(x['Description']) == 'nan' else x['Description'],  axis = 1)
	df_CF = df_CF.reset_index(drop=True)
	df_CF = df_CF[df_CF['Description'].str.contains(keyword)]
	df_CF = df_CF.reset_index(drop=True)

	df_CF_concat=(df_CF.groupby('Date',as_index=False).agg(lambda x:list(set(x))[0] if len(set(x))==1 else list(set(x))))
	df_CF_concat = df_CF_concat[:616]
	df_CF_concat = df_CF_concat.reset_index(drop=True)

	return df_CF_concat


def Torch_list(df_input):
	torch_list = []
	for num in range(len(df_input)):
		X = df_input.embedding_Description[num]
		if type(X) == str:
			X = X.replace('[','')
			X = X.replace(']','')
			X = X.replace('\n','')
			X = np.fromstring(X, dtype=float, sep=' ')

			X = np.reshape(X, (-1, 384))
			X = np.array(X).astype("float32").reshape(1, 1, 384)
			X = torch.tensor(X)
			Y = X
			torch_list.append(Y)
		else:
			X = df_input.embedding_Description[num]

			for i in range(len(X)):
				X[i] = str(X[i])
				X[i] = X[i].replace('[','')
				X[i] = X[i].replace(']','')
				X[i] = X[i].replace('\n','')
				if len(X[i]) <= 5:
					continue
				else:
					X[i] = np.fromstring(X[i], dtype=float, sep=' ')
					X[i] = np.reshape(X[i], (-1, 384))

			if type(X[0]) == str:
				Y = X[1]
				count = len(X) - 1
				for i in range(2,len(X)):
					if type(X[i]) == str:
						count -= 1
					else:
						Y = np.concatenate((Y, X[i]), axis=0)

				Y = np.array(Y).astype("float32").reshape(1, count, 384)
				Y = torch.tensor(Y)    
				torch_list.append(Y)

			else:
				Y = X[0]
				count = len(X)
				for i in range(1,len(X)):
					if type(X[i]) == str:
						count -= 1
					else:
				  		Y = np.concatenate((Y, X[i]), axis=0)
				Y = np.array(Y).astype("float32").reshape(1, count, 384)
				Y = torch.tensor(Y)    
				torch_list.append(Y)

	return torch_list


def Torch_list2(df_CF_concat):
	torch_list = []
	for num in range(len(df_CF_concat)):
		X = df_CF_concat.embedding_Description[num]
		if type(X) == str:
			#一維陣列
			#X = df_CF_concat.embedding_Summary[num].replace('[','')
			X = X.replace('[','')
			X = X.replace(']','')
			X = X.replace('\n','')
			X = np.fromstring(X, dtype=float, sep=' ')

			X = np.reshape(X, (-1, 384))
			X = np.array(X).astype("float32").reshape(1, 1, 384)
			X = torch.tensor(X)
			Y = X
			torch_list.append(Y)
		else:
			#多維陣列
			X = df_CF_concat.embedding_Description[num]

			for i in range(len(X)):
				X[i] = str(X[i])
				X[i] = X[i].replace('[','')
				X[i] = X[i].replace(']','')
				X[i] = X[i].replace('\n','')
				if len(X[i]) <= 5:
					continue
				else:
					X[i] = np.fromstring(X[i], dtype=float, sep=' ')
					X[i] = np.reshape(X[i], (-1, 384))

			if type(X[0]) == str:
				Y = X[1]
				count = len(X) - 1
				for i in range(2,len(X)):
					if type(X[i]) == str:
						count -= 1
					else:
						Y = np.concatenate((Y, X[i]), axis=0)

				Y = np.array(Y).astype("float32").reshape(1, count, 384)
				Y = torch.tensor(Y)    
				torch_list.append(Y)

			else:
				Y = X[0]
				count = len(X)
				for i in range(1,len(X)):
					if type(X[i]) == str:
						count -= 1
					else:
				  		Y = np.concatenate((Y, X[i]), axis=0)
				Y = np.array(Y).astype("float32").reshape(1, count, 384)
				Y = torch.tensor(Y)    
				torch_list.append(Y)
	return torch_list


class Website():

	def __init__(self, ddate1, ddate2):
		self.ddate1 = ddate1
		self.ddate2 = ddate2


	def scrape(self):
		pass


class Website2():

	def __init__(self, source1, source2, first, continued, ddate1):
		self.source1 = source1
		self.source2 = source2
		self.first = first
		self.continued = continued
		self.ddate1 = ddate1


class Website3():

	def __init__(self, source1, source2, first, continued, keyword):
		self.source1 = source1
		self.source2 = source2
		self.first = first
		self.continued = continued
		self.keyword = keyword


class ML_Home(Website2):

	def ML(self):
		try:
			if self.source1 and self.source2 and self.ddate1:
				# print('CNBC+FORBES')
				df_CF_concat = CF()
			elif self.source1 and self.ddate1:
				# print('CNBC')
				df_CF_concat = CNBC()
			elif self.source2 and self.ddate1:
				# print('FORBES')
				df_CF_concat = FORBES()

			for i in range(len(df_CF_concat)):
				if df_CF_concat.Date[i] == self.ddate1:
					x = i-7
					y = i
			try:
				df_input = df_CF_concat.iloc[x:y]
				df_input = df_input.reset_index(drop=True)
			except:
				pass

			torch_list = Torch_list(df_input)

			model = RNN()

			if self.first:
				# print('first')
				df_F_unemployment,  former_list = f_label()
			elif self.continued:
				df_F_unemployment,  former_list = c_label()

			df_former = former_list[-1:]
			outputs = model(torch_list[0],torch_list[1],torch_list[2],torch_list[3],torch_list[4],torch_list[5],torch_list[6],df_former[0])
			predicted = outputs.gt(0.5)
			predicted = int(predicted.int())
			

			result_list = []
			if predicted == 1:
				result = "上漲"
			else:
				result = "下跌"

			result_list.append(dict(result=result))

			return result_list

		except:
			result_list = []
			result = "請輸入正確格式資料"
			result_list.append(dict(result=result))

			return result_list


class Model_Training(Website3):

	def Model(self):
		result_list = []
		if self.source1 and self.source2:
			source = 'CNBC+FORBES'
			if self.keyword:
				# print(self.keyword)
				df_CF_concat = CF_K(self.keyword)
			else:
				df_CF_concat = CF()
			# print(source)

		elif self.source1:
			source = 'CNBC'
			if self.keyword:
				# print(self.keyword)
				df_CF_concat = CNBC_K(self.keyword)
			else:
				df_CF_concat = CNBC()
			# print(source)

		elif self.source2:
			source = 'FORBES'
			if self.keyword:
				# print(self.keyword)
				df_CF_concat = FORBES_K(self.keyword)
			else:
				df_CF_concat = FORBES()
			# print(source)

		else:
			# print('pass')
			return result_list

		if self.first:
			# print('first')
			df_F_unemployment, former_list = f_label()
		elif self.continued:
			# print('continued')
			df_F_unemployment, former_list = c_label()
		else:
			# print('pass')
			return result_list

		try:
			keyword = self.keyword
			
			label_list = []
			dddate_list = []
			for i in range(len(df_F_unemployment)):
				if df_F_unemployment.announce[i] >= df_F_unemployment.former[i]:
				  data = 1
				  label_list.append(data)
				  dddate_list.append(df_F_unemployment.ddate[i])
				else:
				  data = 0
				  label_list.append(data)
				  dddate_list.append(df_F_unemployment.ddate[i])

			torch_list = Torch_list2(df_CF_concat)

			input_list = []
			for i in range(0,len(torch_list),7):
				temp_list = []
				temp_list.append(torch_list[i])
				temp_list.append(torch_list[i+1])
				temp_list.append(torch_list[i+2])
				temp_list.append(torch_list[i+3])
				temp_list.append(torch_list[i+4])
				temp_list.append(torch_list[i+5])
				temp_list.append(torch_list[i+6])
				input_list.append(temp_list)


			df_XY = pd.DataFrame({'input':input_list,'history':former_list, 'label':label_list, 'ddate':dddate_list})
			yy = df_XY['label']
			XX = df_XY.drop(['label'],axis=1)




			model = RNN()

			loss_fn = nn.BCELoss()
			optimizer = opt.SGD(model.parameters(),lr=0.001)
			X_train, X_test, y_train, y_test = train_test_split(XX, yy, test_size=0.3, random_state=0)

			X_train = X_train.reset_index(drop=True)
			X_test = X_test.reset_index(drop=True)
			y_train = y_train.reset_index(drop=True)
			y_test = y_test.reset_index(drop=True)

			correct = 0
			total = 0
			j = 0

			result_list = []

			for i in range (len(X_test)):
				outputs = model(X_test['input'][i][j],X_test['input'][i][j+1],X_test['input'][i][j+2],X_test['input'][i][j+3],X_test['input'][i][j+4],X_test['input'][i][j+5],X_test['input'][i][j+6],X_test['history'][i])
				predicted = outputs.gt(0.5)
				correct += (predicted.int() == y_test[i]).sum()
				predicted=int(predicted.int())
				if predicted == 1:
				  	predicted = '上漲'
				else:
				  	predicted = '下跌'

				test = y_test[i]
				if test == 1:
				  	test = '上漲'
				else:
				  	test = '下跌'

				result_list.append(dict(predicted=predicted, y_test=test, ddate = X_test['ddate'][i], keyword=keyword, source=source))

			total = len(y_test)
			accuracy = "%.3f%%"%(100.0 * float(correct)//float(total))
			accuracy = str(accuracy.split('.')[0])

			if int(accuracy) < 50:
				result_list = []
				correct = 0
				total = 0
				for i in range (len(X_test)):
					outputs = model(X_test['input'][i][j],X_test['input'][i][j+1],X_test['input'][i][j+2],X_test['input'][i][j+3],X_test['input'][i][j+4],X_test['input'][i][j+5],X_test['input'][i][j+6],X_test['history'][i])
					predicted = outputs.gt(0.5)
					predicted = int(predicted.int())
					if predicted == 0:
					  	predicted = 1
					else:
					  	predicted = 0
					
					correct += (predicted == y_test[i]).sum()

					if predicted == 0:
					  	predicted = '下跌'
					else:
					  	predicted = '上漲'

					test = y_test[i]
					if test == 1:
					  	test = '上漲'
					else:
					  	test = '下跌'

					result_list.append(dict(predicted=predicted, y_test=test, ddate = X_test['ddate'][i], keyword=keyword, source=source))

				total = len(y_test)
				accuracy = "%.3f%%"%(100.0 * float(correct)//float(total))
				accuracy = str(accuracy.split('.')[0])

			result_list.append(dict(accuracy=accuracy))


			return result_list

		except:
			return result_list


class First_unemployed(Website):

	def scrape(self):
		df = pd.read_excel(r'/home/victor/MIS_finalproject/美國初領失業救濟金人數.xlsx')
		df.ddate = df.ddate.str.replace(" ","")

		ddate_list = []
		announce_list = []
		former_list = []

		result_list = []
		if self.ddate1 and self.ddate2:
			for i in range(len(df)):
				if df.ddate[i] == self.ddate2:
					x = i
				if df.ddate[i] == self.ddate1:
					y = i
			try:
				df = df.iloc[x:y+1]
				df = df.reset_index(drop=True)

				for i in range(len(df)):
					ddate_list.append(df.ddate[i])
					announce_list.append(df.announce[i])
					former_list.append(df.former[i])

					result_list.append(dict(ddate=df.ddate[i], announce=df.announce[i].replace('K','000'), former=df.former[i].replace('K','000')))
				
				return result_list

			except:

				pass

		else:

			for i in range(len(df)):
				ddate_list.append(df.ddate[i])
				announce_list.append(df.announce[i])
				former_list.append(df.former[i])
				
				result_list.append(dict(ddate=df.ddate[i], announce=df.announce[i].replace('K','000'), former=df.former[i].replace('K','000')))

			return result_list	


class Continue_unemployed(Website):
	
	def scrape(self):
		df = pd.read_excel(r'/home/victor/MIS_finalproject/美國續領失業救濟金人數.xlsx')
		df.ddate = df.ddate.str.replace(" ","")

		ddate_list = []
		announce_list = []
		former_list = []

		result_list = []

		if self.ddate1 and self.ddate2:
			for i in range(len(df)):
				if df.ddate[i] == self.ddate2:
					x = i
				if df.ddate[i] == self.ddate1:
					y = i
					
			try:
				df = df.iloc[x:y+1]
				df = df.reset_index(drop=True)

				for i in range(len(df)):
					ddate_list.append(df.ddate[i])
					announce_list.append(df.announce[i])
					former_list.append(df.former[i])

					result_list.append(dict(ddate=df.ddate[i], announce=df.announce[i].replace('K','000'), former=df.former[i].replace('K','000')))
				
				return result_list

			except:
				
				pass

		else:

			for i in range(len(df)):
				ddate_list.append(df.ddate[i])
				announce_list.append(df.announce[i])
				former_list.append(df.former[i])

				result_list.append(dict(ddate=df.ddate[i], announce=df.announce[i].replace('K','000'), former=df.former[i].replace('K','000')))
			
			return result_list

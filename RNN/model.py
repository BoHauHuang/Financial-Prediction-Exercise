import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Models(object):
	def __init__(self, data, train_col, test_col, normalize=True, test_size=0.01):
		self.data = data
		self.train_col = train_col
		self.test_col = test_col
		self.test_size = test_size
		self.past_period = 14
		self.timedelay = 1

		self.X = []
		self.Y = []

		self.x_train = []
		self.y_train = []
		self.x_test = []
		self.y_test = []

		self.future_x = []
		self.future_price = 0

		self.train_predict = []
		self.test_predict = []

		self.RNN_model = None

		if normalize:
			self.normalize()

		self.construct_train()
		self.train_test_split()

	def normalize(self):
		self.data = self.data.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
		print(len(self.data))
	def construct_train(self):
		x = []
		y = []
		future = []
		for i in range(0, self.data.shape[0]-self.past_period+1):
			if i == self.data.shape[0]-self.past_period:
				future.append(np.array(self.data.iloc[i : i+self.past_period][self.train_col]))
			else:
				x.append(np.array(self.data.iloc[i : i+self.past_period][self.train_col]))

		for i in range(self.past_period, self.data.shape[0]-self.timedelay+1):
			y.append(np.array(self.data.iloc[i:i+self.timedelay][self.test_col]))

		self.X = np.array(x)
		self.Y = np.array(y)
		self.future_x = np.array(future)

	def train_test_split(self):
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=self.test_size, shuffle=False)
		self.y_train = self.y_train[:,0]
		self.y_test = self.y_test[:,0]

	def build_RNN_model(self):
		# Many to one model
		self.RNN_model = Sequential()
		self.RNN_model.add(LSTM(self.past_period, input_shape=(self.x_train.shape[1], self.x_train.shape[2]), return_sequences=True))
		#self.RNN_model.add(Dropout(0.1))
		self.RNN_model.add(LSTM(self.past_period, return_sequences=True))
		#self.RNN_model.add(Dropout(0.1))
		self.RNN_model.add(LSTM(self.past_period))
		self.RNN_model.add(Dense(1))
		self.RNN_model.compile(loss="mean_squared_error", optimizer="adam")
		self.RNN_model.summary()

	def fit_RNN_model(self):
		callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
		self.RNN_model.fit(self.x_train, self.y_train, epochs=100, batch_size=128, validation_data=(self.x_test, self.y_test), callbacks=[callback])

	def predict_RNN_model(self):
		
		self.train_predict = self.RNN_model.predict(self.x_train)
		self.test_predict = self.RNN_model.predict(self.x_test)

		return self.train_predict, self.test_predict
	
	def plot_img(self, train_predict, test_predict):
		train_Score = math.sqrt(mean_squared_error(self.y_train[:,0], train_predict[:,0]))
		print('Train Score: %.2f Error' %(train_Score))
		testScore = math.sqrt(mean_squared_error(self.y_test[:,0], test_predict[:,0]))
		print('Test Score: %.2f Error' % (testScore))
		
		trainPredictPlot = np.empty_like(self.Y[:,0])
		trainPredictPlot[:, :] = np.nan
		train_len = len(train_predict[:])
		trainPredictPlot[ : train_len] = train_predict[:]

		testPredictPlot = np.empty_like(self.Y[:,0])
		testPredictPlot[:, :] = np.nan
		test_len = len(test_predict[:])
		testPredictPlot[-test_len: ] = test_predict[:]

		plt.plot(self.Y[:,0])
		plt.plot(trainPredictPlot,'r')
		plt.plot(testPredictPlot,'b')


		self.future_price = self.RNN_model.predict(self.future_x)
		x_ = len(self.Y[:,0])
		y_ = self.future_price
		plt.plot(x_, y_, 'ro')

		'''
		plt.plot(self.y_test[:-2], 'r')
		plt.plot(self.test_predict[1:], 'b')
		'''
		plt.show()
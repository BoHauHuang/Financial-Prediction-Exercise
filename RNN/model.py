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
		self.past_period = 7
		self.timedelay = 1

		self.X = []
		self.Y = []

		self.x_train = []
		self.y_train = []
		self.x_test = []
		self.y_test = []

		self.RNN_model = None

		if normalize:
			self.normalize()

		self.construct_train()
		self.train_test_split()

	def normalize(self):
		self.data = self.data.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

	def construct_train(self):
		x = []
		y = []
		for i in range(self.data.shape[0]-self.timedelay-self.past_period):
			x.append(np.array(self.data.iloc[i:i+self.past_period][self.train_col]))
			y.append(np.array(self.data.iloc[i+self.past_period:i+self.past_period+self.timedelay][self.test_col]))

		self.X = np.array(x)
		self.Y = np.array(y)

	def train_test_split(self):
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=self.test_size, shuffle=False)
		self.y_train = self.y_train[:,0]
		self.y_test = self.y_test[:,0]

	def build_RNN_model(self):
		# Many to one model
		self.RNN_model = Sequential()
		self.RNN_model.add(LSTM(10, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
		self.RNN_model.add(Dense(1))
		self.RNN_model.compile(loss="mean_squared_error", optimizer="adam")
		self.RNN_model.summary()

	def fit_RNN_model(self):
		callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
		self.RNN_model.fit(self.x_train, self.y_train, epochs=1000, batch_size=128, validation_data=(self.x_test, self.y_test), callbacks=[callback])

	def predict_RNN_model(self):
		
		train_predict = self.RNN_model.predict(self.x_train)
		test_predict = self.RNN_model.predict(self.x_test)

		train_Score = math.sqrt(mean_squared_error(self.y_train[:,0], train_predict[:,0]))
		print('Train Score: %.2f Error' %(train_Score))
		testScore = math.sqrt(mean_squared_error(self.y_test[:,0], test_predict[:,0]))
		print('Test Score: %.2f Error' % (testScore))

		trainPredictPlot = np.empty_like(self.data)
		trainPredictPlot[:, :] = np.nan
		trainPredictPlot[self.past_period-self.timedelay:len(train_predict)+self.past_period-self.timedelay] = train_predict

		testPredictPlot = np.empty_like(self.data[self.test_col])
		testPredictPlot[:] = np.nan
		testPredictPlot[len(train_predict)+self.past_period-2*self.timedelay:len(self.data[self.test_col])-3*self.timedelay] = test_predict

		plt.plot(self.data[self.test_col].values)
		plt.plot(trainPredictPlot,'r')
		plt.plot(testPredictPlot,'b')
		plt.show()
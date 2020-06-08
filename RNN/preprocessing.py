import talib

class Indicator(object):
	def __init__(self, data):
		self.data = data

	def RSI(self, timeperiod=14):
		return talib.RSI(self.data.Close, timeperiod=timeperiod)

	def MACD(self, fast=12, slow=26, signalperiod=9):
		return talib.MACD(self.data.Close, fastperiod=fast, slowperiod=slow, signalperiod=signalperiod)

	def MA(self, timeperiod=30, matype=0):
		return talib.MA(self.data.Close, timeperiod=timeperiod, matype=matype)

	def STDDEV(self, timeperiod=5, nbdev=1):
		return talib.STDDEV(self.data.Close, timeperiod=timeperiod, nbdev=nbdev)

	def BBands(self, confidence=2):
		# confidence=2: 95%
		return self.MA(20)+confidence*self.STDDEV(20), self.MA(20), self.MA(20)-confidence*self.STDDEV(20)

	def MFI(self, timeperiod=14):
		return talib.MFI(self.data.High, self.data.Low, self.data.Close, self.data.Volume, timeperiod=timeperiod)

	def Trend(self, timeperiod=7):
		trend = [0]
		for i in range(1, timeperiod):
			if self.data.Close[i] <= self.data.Close[i-1]:
				trend.append(0)
			else:
				trend.append(1)

		for i in range(timeperiod, len(self.data.Close)):
			if self.data.Close[i] <= self.data.Close[i-timeperiod]:
				trend.append(0)
			else:
				trend.append(1)

		return trend

	def Slope(self, timeperiod=7):
		return talib.LINEARREG_SLOPE(self.data.Close, timeperiod=timeperiod)
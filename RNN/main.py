from collecter import DataCollecter
from plot_candle import Plot
from preprocessing import Indicator

url = 'https://finance.yahoo.com/quote/%5EGSPC?p=^GSPC'
filepath = './SP500.csv'

#----  Data Collect  ----
Collecter = DataCollecter(url, filepath)

data = Collecter.load_data_from_fred('SP500')
history = Collecter.load_history_from_yahoo('^GSPC')[['Open', 'Close', 'Low', 'High', 'Volume']]

#----  Indicator ----
indicator = Indicator(history[:])

macd, macd_signal, macd_histogram = indicator.MACD(fast=12, slow=26, signalperiod=9)
rsi = indicator.RSI(timeperiod=14)
ma = indicator.MA(timeperiod=14, matype=0)
b_high, b_mid, b_low = indicator.BBands()
mfi = indicator.MFI(timeperiod=14)
slope = indicator.Slope(timeperiod=7)
#trend = indicator.Trend(timeperiod=5)

history['MACD'] = macd
history['MACD_SIG'] = macd_signal
history['MACD_HIST'] = macd_histogram
history['RSI'] = rsi
history['MA'] = ma
history['BBAND_HIGH'] = b_high
history['BBAND_MID'] = b_mid
history['BBAND_LOW'] = b_low
history['MFI'] = mfi
history['Slope'] = slope
#history['Trend'] = trend

history.to_csv(filepath, encoding='utf-8')

#----  Plot  ----
#Plot.plot_candles(history[:], technicals=[rsi], technicals_titles=['RSI'])

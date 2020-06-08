import numpy as np
import pandas as pd
from model import Models

needed = ["Open", "High", "Low", "Volume", "MACD", "MACD_SIG", "MACD_HIST", "RSI", "MA", "BBAND_HIGH", "BBAND_MID", "BBAND_LOW", "MFI", "Slope", "Close"]
train_col = ["Open", "High", "Low", "Volume", "MACD", "MACD_SIG", "MACD_HIST", "RSI", "MA", "BBAND_HIGH", "BBAND_MID", "BBAND_LOW", "MFI", "Slope"]
test_col = ["Close"]

data = pd.read_csv("./SP500.csv", encoding='utf-8')
data = data.dropna(axis=0, how='any')
data = data[needed]

model = Models(data, train_col, test_col)

model.build_RNN_model()
model.fit_RNN_model()
model.predict_RNN_model()
#output = pd.DataFrame([])
#output["real"] = y_test
#output["predict"] = predict

#output.to_csv('result.csv', encoding='utf-8')
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
train_predict, test_predict = model.predict_RNN_model()
model.plot_img(train_predict, test_predict)

test_x = pd.DataFrame(model.x_test[:,0][:-2])
test_x.columns = train_col
test_y = pd.DataFrame(model.y_test[:-2])
test_y.columns = test_col
predict = pd.DataFrame(test_predict[1:-1])
predict.columns = ["Predict"]
result = pd.concat([test_x, test_y, predict], axis=1)
result.to_csv("result.csv", encoding='utf-8')
#output = pd.DataFrame([])
#output["real"] = y_test
#output["predict"] = predict

#output.to_csv('result.csv', encoding='utf-8')
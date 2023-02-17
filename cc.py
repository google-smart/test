import requests
import pandas as pd
import numpy as np
import talib
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 定义API接口URL
url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h"

# 获取历史数据
res = requests.get(url)
data = res.json()

# 将数据转换为DataFrame格式
df = pd.DataFrame(data, columns=["Open time", "Open", "High", "Low", "Close", "Volume",
                                 "Close time", "Quote asset volume", "Number of trades",
                                 "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"])

# 将时间戳转换为可读时间
df["Open time"] = pd.to_datetime(df["Open time"], unit='ms')

# 将字符串类型的数据转换为浮点类型
df[["Open", "High", "Low", "Close", "Volume"]] = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

# 计算MA指标
df["MA5"] = talib.MA(df["Close"], timeperiod=5)
df["MA10"] = talib.MA(df["Close"], timeperiod=10)
df["MA20"] = talib.MA(df["Close"], timeperiod=20)

# 去除不需要的列
df.drop(columns=["Open time", "Close time", "Quote asset volume", "Number of trades",
                  "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"], inplace=True)

# 去除包含NaN的行
df.dropna(inplace=True)

# 将数据归一化
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 将数据划分为训练集和测试集
X = df_scaled.iloc[:, :-1].values
y = df_scaled.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 将数据转换为3D格式，以符合LSTM网络输入格式
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32)

# 测试模型
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {test_loss}")

# 保存模型
model.save("btc_model.h5")

# 保存归一化器
with open("scaler.pkl", "wb") as f:

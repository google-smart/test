import requests
import pandas as pd
import numpy as np
import talib
import pickle
from keras.models import load_model
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
    pickle.dump(scaler, f)

print("Scaler saved as scaler.pkl")

# 加载模型和归一化器
model = load_model("btc_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 获取最新K线数据
res = requests.get(url)
data = res.json()
df_latest = pd.DataFrame(data, columns=["Open time", "Open", "High", "Low", "Close", "Volume",
                                        "Close time", "Quote asset volume", "Number of trades",
                                        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"])
df_latest["Open time"] = pd.to_datetime(df_latest["Open time"], unit='ms')
df_latest[["Open", "High", "Low", "Close", "Volume"]] = df_latest[["Open", "High", "Low", "Close", "Volume"]].astype(float)
df_latest.drop(columns=["Open time", "Close time", "Quote asset volume", "Number of trades",
                         "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"], inplace=True)

# 计算最新K线数据的MA指标
df_latest["MA5"] = talib.MA(df_latest["Close"], timeperiod=5)
df_latest["MA10"] = talib.MA(df_latest["Close"], timeperiod=10)
df_latest["MA20"] = talib.MA(df_latest["Close"], timeperiod=20)

# 去除包含NaN的行
df_latest.dropna(inplace=True)

# 将最新K线数据归一化
df_latest_scaled = pd.DataFrame(scaler.transform(df_latest), columns=df_latest.columns)

# 将最新K线数据转换为3D格式
X_latest = df_latest_scaled.iloc[:, :-1].values
X_latest = X_latest.reshape(X_latest.shape[0], X_latest.shape[1], 1)

# 预测最新K线数据的涨跌趋势
y_latest_pred = model.predict(X_latest)
y_latest_pred = np.round(scaler.inverse_transform(y_latest_pred)[-1][0], 2)

# 计算最新K线数据的涨跌幅度
latest_close = df_latest.iloc[-1]["Close"]
change_percent = (y_latest_pred - latest_close) / latest_close * 100

print(f"Latest close price: {latest_close:.2f}")
print(f"Predicted close price: {y_latest_pred:.2f}")
print(f"Change percentage: {change_percent:.2f}%")

# 将最新K线数据的涨跌趋势和涨跌幅度保存到文件中
with open("latest_prediction.txt", "w") as f:
    f.write(f"Latest close price: {latest_close:.2f}\n")
    f.write(f"Predicted close price: {y_latest_pred:.2f}\n")
    f.write(f"Change percentage: {change_percent:.2f}%\n")

print("Prediction saved as latest_prediction.txt")

# 读取数据并将数据拆分为特征和目标变量
features, targets = read_data()

# 对数据进行标准化处理
features = standardize_data(features)

# 训练模型
model = train_model(features, targets)

# 预测目标变量
predictions = predict(model, features)

# 计算并打印性能指标
accuracy = compute_accuracy(predictions, targets)
print(f"Accuracy: {accuracy}")

# 读取数据并将数据拆分为特征和目标变量
features, targets = read_data()

# 对数据进行标准化处理
features = standardize_data(features)

# 交叉验证评估模型
scores = cross_val_score(model, features, targets, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean()}")
# 计算并打印其他性能指标，如精确度、召回率、F1得分等
precision = compute_precision(predictions, targets)
recall = compute_recall(predictions, targets)
f1_score = compute_f1_score(predictions, targets)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 score: {f1_score}")


# 定义参数网格，以便对超参数进行网格搜索
param_grid = {
    "alpha": [0.01, 0.1, 1.0, 10.0],
    "hidden_layer_sizes": [(10,), (50,), (100,), (10, 10), (50, 50), (100, 100)]
}

# 使用网格搜索调整超参数
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(train_data, train_labels)

# 打印最佳参数组合
print("Best parameters:", grid_search.best_params_)

# 使用最佳参数训练模型
best_model = MLPRegressor(**grid_search.best_params_)
best_model.fit(train_data, train_labels)



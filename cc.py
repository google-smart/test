import requests
import pandas as pd
import numpy as np
import talib
import pickle
from keras.models import load_model
import pandas as pd
import numpy as np
import talib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from ta import add_all_ta_features
import requests
import pandas as pd

# 定义API接口及参数
url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
params = {'vs_currency': 'usd', 'days': '365'}

# 获取数据并转换为DataFrame格式
response = requests.get(url, params=params)
data = response.json()
df = pd.DataFrame(data['prices'], columns=['date', 'price'])
df['date'] = pd.to_datetime(df['date'], unit='ms')
df.set_index('date', inplace=True)
print(df['date'].head())

# 添加技术指标
assert set(['open', 'high', 'low', 'close', 'volume']).issubset(set(df.columns)), "DataFrame is missing required columns"
df = add_all_ta_features(df, "open", "high", "low", "close", "volume", fillna=True)
df = df.dropna()


# 准备特征和标签
X = df.drop(columns=['close'])
y = np.where(df['close'].shift(-1) > df['close'], 1, 0)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. 模型训练和评估
# 定义模型
def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 定义模型评估函数
def evaluate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"CV scores: {scores}")
    print(f"CV accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# 使用随机森林进行交叉验证
rf = RandomForestClassifier(random_state=42)
evaluate_model(rf, X_train, y_train)

# 使用XGBoost进行交叉验证和超参数调优
pipe = Pipeline([
    ('scale', StandardScaler()),
    ('xgb', XGBClassifier(random_state=42))
])
params = {
    'xgb__learning_rate': [0.01, 0.1],
    'xgb__max_depth': [3, 5, 7],
    'xgb__n_estimators': [100, 200, 300],
}
search = GridSearchCV(pipe, param_grid=params, cv=5, n_jobs=-1, scoring='accuracy')
# Fit the model
search.fit(X_train, y_train)

# Get the best model
best_model = search.best_estimator_

# Print the best parameters and score
print(f"Best parameters: {search.best_params_}")
print(f"Best score: {search.best_score_}")

# Use the best model to make predictions on the test set
y_pred = best_model.predict(X_test)

# Output model performance
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


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


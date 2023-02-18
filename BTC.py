import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

## 1.数据准备
# 读取历史价格数据
data = pd.read_csv("./Data/BTC-USD2.csv")

# 将 Date 列作为索引
data = pd.read_csv('BTC-USD.csv', index_col='Date', parse_dates=['Date'])

# 将日期转换为时间戳并删除 Adj Close 列
data.index.name = 'Timestamp'
data = data.reset_index()
data = data.drop(['Date', 'Adj Close'], axis=1)
data['Timestamp'] = data['Timestamp'].astype('int64') // 10**9

# 选取收盘价作为预测目标
target_col = 'Close'

# 提取特征和目标向量
features = data.drop(target_col, axis=1)
target = data[[target_col]]

# 划分训练集和测试集
train_size = int(len(data) * 0.8)
train_features = features[:train_size]
train_target = target[:train_size]
test_features = features[train_size:]
test_target = target[train_size:]

## 2.特征工程
# 对数据进行归一化处理
scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_features)
train_target = scaler.fit_transform(train_target)
test_features = scaler.fit_transform(test_features)
test_target = scaler.fit_transform(test_target)

## 3.模型选择
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(128, input_shape=(train_features.shape[1], 1), return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1))

## 4.模型训练
model.compile(optimizer='adam', loss='mse')
model.fit(train_features, train_target, epochs=50, batch_size=32, verbose=2)

## 5.模型评估
from sklearn.metrics import mean_squared_error

# 在测试集上进行预测
test_predictions = model.predict(test_features)

# 反归一化处理
test_predictions = scaler.inverse_transform(test_predictions)
test_target = scaler.inverse_transform(test_target)

# 计算均方根误差
mse = mean_squared_error(test_target, test_predictions)
rmse = np.sqrt(mse)
print(f"测试集上的均方根误差为：{rmse:.2f}")

## 6.参数调优：使用Keras Tuner自动化调参
# 定义超参数搜索空间
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('unit', min_value=32, max_value=512, step=32),
                   return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(LSTM(units=hp.Int('unit_' + str(i), min_value=32, max_value=512, step=32), return_sequences=True))
        model.add(Dropout(0.2))
    model.add(LSTM(units=hp.Int('unit_last', min_value=32, max_value=512, step=32)))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error')

    return model

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='btc_prediction')

# 开始搜索最佳超参数
tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 输出搜索结果
tuner.results_summary()

## 7.预测应用：使用训练好的模型对新数据进行预测
# 使用训练好的模型对新数据进行预测
new_data = pd.read_csv('./Data/BTC-USD.csv')
new_data = new_data.set_index('time')
new_data.index = pd.to_datetime(new_data.index, unit='s')
new_data = new_data[['close']]
new_data = new_data.dropna()

scaler = MinMaxScaler(feature_range=(0, 1))
new_data = scaler.fit_transform(new_data)

X_new, y_new = [], []
for i in range(window_size, len(new_data)):
    X_new.append(new_data[i-window_size:i, 0])
    y_new.append(new_data[i, 0])

X_new, y_new = np.array(X_new), np.array(y_new)
X_new = np.reshape(X_new, (X_new.shape[0], X_new.shape[1], 1))

predicted_price = model.predict(X_new)
predicted_price = scaler.inverse_transform(predicted_price)

# 可视化预测结果
plt.plot(new_data[window_size:], color='blue', label='Actual BTC Price')
plt.plot(predicted_price, color='green', label='Predicted BTC Price')
plt.title('BTC Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

## 8.自动化学习：使用Keras Tuner自动化学习调整模型的参数和结构
# 自动化学习调整模型的参数和结构
tuner.search_space_summary()
tuner.results_summary()
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

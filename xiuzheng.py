import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras_tuner as kt

# 1. 数据准备
df = pd.read_csv('https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv')
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']
df = df.sort_index(ascending=True, axis=0)
data = df.filter(['Close'])
dataset = data.values
training_data_len = int(np.ceil( len(dataset) * .8 ))

# 2. 特征工程
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# 3. 模型选择
def model_builder(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   return_sequences=True, input_shape=(scaled_data.shape[1], 1)))
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
    return model

# 4. 模型训练
X_train = []
y_train = []
for i in range(60, len(scaled_data)):
    X_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

tuner = kt.RandomSearch(model_builder,
                        objective='val_loss',
                        max_trials=5,
                        executions_per_trial=3,
                        directory='test_dir',
                        project_name='helloworld')
tuner.search_space_summary()

# 5. 模型评估
X_test = []
y_test = []
for i in range(len(scaled_data)-60, len(scaled_data)):
    X_test.append(scaled_data[i-60:i, 0])
    y_test.append(scaled_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

best_model = tuner.get_best_models(num_models=1)[0]
y_test_predicted = best_model.predict(X_test)
y_test_predicted = scaler.inverse_transform(y_test_predicted)
mse = ((y_test - y_test_predicted) ** 2).mean()
print(f"MSE: {mse}")

# 6. 参数调优
tuner = RandomSearch(
    build_model,
    objective='val_mse',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='helloworld'
)

tuner.search(X_train_scaled, y_train, epochs=20, validation_split=0.2)

# 7. 最佳模型评估
best_model = tuner.get_best_models(num_models=1)[0]
y_test_predicted = best_model.predict(X_test_scaled)
mse = ((y_test - y_test_predicted) ** 2).mean()
print(f"MSE: {mse}")

# 8. 模型保存
best_model.save("my_model.h5")

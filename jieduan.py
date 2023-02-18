import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras_tuner.tuners import RandomSearch

# 1. 数据准备
df = pd.read_csv("data.csv")
df = df[["open", "high", "low", "close", "volume"]]

split_percent = 0.8
split = int(split_percent * len(df))

train_data = df[:split]
test_data = df[split:]

# 2. 特征工程
scaler = MinMaxScaler()

X_train = train_data.drop("close", axis=1)
y_train = train_data[["close"]]
X_test = test_data.drop("close", axis=1)
y_test = test_data[["close"]]

X_train_scaled = scaler.fit_transform(X_train)
y_train_scaled = scaler.fit_transform(y_train)
X_test_scaled = scaler.fit_transform(X_test)
y_test_scaled = scaler.fit_transform(y_test)

X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# 3. 模型选择
def build_model(hp):
    model = keras.Sequential()

    model.add(layers.LSTM(units=hp.Int('unit', min_value=32, max_value=512, step=32), input_shape=(X_train_scaled.shape[1], 1)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(units=hp.Int('dense', min_value=32, max_value=512, step=32), activation=hp.Choice('dense_activation', values=['relu', 'tanh', 'sigmoid', 'linear'])))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(1))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error',
                  metrics=['mse'])

    return model

# 6. 参数调优
tuner = RandomSearch(
    build_model,
    objective='val_mse',
    max_trials=5,
    executions_per_trial=3,
    directory='my_dir',
    project_name='helloworld')

tuner.search(X_train_scaled, y_train_scaled, epochs=20, validation_split=0.2)

# 4. 模型训练
model = tuner.get_best_models(num_models=1)[0]

model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, validation_split=0.1)

# 5. 模型评估
y_test_scaled_predicted = model.predict(X_test_scaled)

y_test_predicted = scaler.inverse_transform(y_test_scaled_predicted)
y_test = np.array(y_test)

mse = ((y_test - y_test_predicted) ** 2).mean()
print(f"MSE: {mse}")
这里被截断了

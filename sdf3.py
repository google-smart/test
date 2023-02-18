import pandas as pd
import numpy as np
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

# 1. 数据准备
df = pd.read_csv('./Data/BTC-USD2.csv')
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

def extract_features(df, column_names):
    data = df[column_names]
    return data.to_numpy()

train_X = extract_features(train_df, ['Open', 'High', 'Low', 'Close', 'Volume'])
test_X = extract_features(test_df, ['Open', 'High', 'Low', 'Close', 'Volume'])
train_y = extract_features(train_df, ['Close'])
test_y = extract_features(test_df, ['Close'])

# 2. 特征工程
scaler = MinMaxScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)
train_y = scaler.fit_transform(train_y)
test_y = scaler.transform(test_y)

# 3. 模型选择
def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), input_shape=(train_X.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

# 4. 模型训练
def train_model(X_train, y_train, X_val, y_val, epochs, batch_size):
    model = build_model(HyperParameters())
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stopping])
    return model

model = train_model(train_X.reshape(train_X.shape[0], train_X.shape[1], 1), train_y,
                    test_X.reshape(test_X.shape[0], test_X.shape[1], 1), test_y, epochs=50, batch_size=64)

# 5. 模型评估
def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)
    mse = np.mean((predictions - y_test)**2)
    print('MSE:', mse)

evaluate_model(model, test_X.reshape(test_X.shape[0], test_X.shape[1], 1), test_y, scaler)

# 6. 参数调优
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=20,
    factor=3,
    directory='my_dir',
    project_name='intro_to_kt')

# 运行超参数搜索
tuner.search(X_train_scaled, y_train, epochs=20, validation_split=0.2)

# 获取最佳模型的超参数
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"The hyperparameters search is complete. The optimal number of units in the first densely-connected layer is {best_hps.get('units')} and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.")

# 根据最佳的超参数训练模型
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train_scaled, y_train, epochs=20, validation_split=0.2)

# 7. 预测应用
# 对新数据进行预测
new_data = pd.read_csv('./Data/BTC-USD.csv')
new_data = new_data.drop('date', axis=1)
new_data = scaler.transform(new_data)
new_data = np.reshape(new_data, (new_data.shape[0], new_data.shape[1], 1))

prediction = model.predict(new_data)
print(f"Predicted price changes: {prediction}")

# 8. 自动化学习
# 使用Keras Tuner自动化学习调整模型的参数和结构
tuner = kt.AutoTuner(
    build_model,
    objective='val_loss',
    max_epochs=10,
    seed=42,
    project_name='intro_to_kt')

# 运行自动化学习
tuner.search(X_train_scaled, y_train, epochs=20, validation_split=0.2)

# 获取最佳模型的超参数
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

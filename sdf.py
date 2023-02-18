import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error

# 3. 模型选择
def lstm_model(input_shape):
    """
    构建LSTM模型
    """
    model = keras.Sequential()
    model.add(layers.LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(50, return_sequences=True))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(50))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1))

    return model

# 4. 模型训练
def train_model(X_train, y_train, X_test, y_test):
    """
    训练LSTM模型
    """
    model = lstm_model((X_train.shape[1], X_train.shape[2]))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    return model, history

# 5. 模型评估
def evaluate_model(model, X_test, y_test, scaler):
    # 将测试数据集进行归一化
    X_test_scaled = scaler.transform(X_test)

    # 对测试数据进行预测
    y_pred_scaled = model.predict(X_test_scaled)

    # 将预测结果反归一化
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # 计算均方误差和平均绝对误差
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print('均方误差：%.2f' % mse)
    print('平均绝对误差：%.2f' % mae)

# 6. 参数调优
def tune_hyperparameters(X_train, y_train, scaler):
    # 构建 KerasRegressor 对象
    model = KerasRegressor(build_fn=create_model, verbose=0)

    # 定义要调整的参数
    param_grid = {'batch_size': [32, 64, 128],
                  'epochs': [10, 20, 30],
                  'optimizer': ['adam', 'rmsprop']}

    # 使用 GridSearchCV 进行参数调优
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    grid_result = grid_search.fit(X_train, y_train)

    # 输出最优参数
    print('最优参数：', grid_result.best_params_)

    # 根据最优参数重新训练模型
    model = create_model(optimizer=grid_result.best_params_['optimizer'])
    model.fit(X_train, y_train, batch_size=grid_result.best_params_['batch_size'], epochs=grid_result.best_params_['epochs'])

    # 对模型进行评估
    evaluate_model(model, X_test, y_test, scaler)

# 7. 预测应用
def predict(model, X, scaler):
    # 预测
    y_pred = model.predict(X)
    # 将预测结果进行反归一化
    y_pred = scaler.inverse_transform(y_pred)
    return y_pred

# 8. 自动化学习
def auto_train(X_train, y_train, X_val, y_val, num_features):
    # 定义Keras Tuner搜索空间
    def model_builder(hp):
        model = Sequential()
        # 添加LSTM层，共搜索三个LSTM层的神经元数
        for i in range(hp.Int('num_lstm_layers', 1, 3)):
            # 第一层需要输入维度信息
            if i == 0:
                model.add(LSTM(units=hp.Int('units_'+str(i), min_value=32, max_value=256, step=32),
                               input_shape=(X_train.shape[1], num_features),
                               return_sequences=True))
            else:
                model.add(LSTM(units=hp.Int('units_'+str(i), min_value=32, max_value=256, step=32),
                               return_sequences=True))
        # 添加全连接层，搜索一个神经元数
        model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=256, step=32)))
        # 添加输出层
        model.add(Dense(units=1))
        # 编译模型
        model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        return model

    # 初始化Keras Tuner
    tuner = kt.Hyperband(
        model_builder,
        objective='val_mean_squared_error',
        max_epochs=10,
        factor=3,
        directory='my_dir',
        project_name='intro_to_kt')

    # 开始搜索最佳模型超参数
    tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    # 获取最佳超参数和最佳模型
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    # 训练最佳模型并返回
    best_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    return best_model

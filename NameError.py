E:\训练模型.py:8: DeprecationWarning: `import kerastuner` is deprecated, please use `import keras_tuner`.
  from kerastuner.tuners import RandomSearch
2023-02-18 15:17:46.374862: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/50
39/39 [==============================] - 1s 7ms/step - loss: 0.0179 - mse: 0.0179 - val_loss: nan - val_mse: nan
Epoch 2/50
39/39 [==============================] - 0s 2ms/step - loss: 9.9602e-04 - mse: 9.9602e-04 - val_loss: nan - val_mse: nan
Epoch 3/50
39/39 [==============================] - 0s 2ms/step - loss: 1.1323e-04 - mse: 1.1323e-04 - val_loss: nan - val_mse: nan
Epoch 4/50
39/39 [==============================] - 0s 2ms/step - loss: 9.2895e-05 - mse: 9.2895e-05 - val_loss: nan - val_mse: nan
Epoch 5/50
39/39 [==============================] - 0s 2ms/step - loss: 8.3015e-05 - mse: 8.3015e-05 - val_loss: nan - val_mse: nan
20/20 [==============================] - 0s 579us/step
MSE: nan
Traceback (most recent call last):
  File "E:\训练模型.py", line 71, in <module>
    tuner.search(X_train_scaled, y_train, epochs=20, validation_split=0.2)
NameError: name 'X_train_scaled' is not defined


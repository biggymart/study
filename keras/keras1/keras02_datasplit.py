# 툴 사용하지 않고 수작업으로 data split한 파일

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. data
x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([1, 2, 3, 4, 5])
x_test = np.array([6, 7, 8])
y_test = np.array([6, 7, 8])

#2. model
model = Sequential()
model.add(Dense(5, input_dim=1, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(4))
model.add(Dense(1))

#3. compile and fit
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. evaluate and predict
loss = model.evaluate(x_test, y_test, batch_size=1)
print('Loss :', loss)

result = model.predict([9])
print('Result :', result)

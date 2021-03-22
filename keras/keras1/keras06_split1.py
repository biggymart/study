# 리스트 슬라이싱을 활용한 data split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. data
x = np.array(range(1, 101))
y = np.array(range(101, 201))

# list slicing
x_train = x[:60] # 1~60, index from 0 to 59
x_val = x[60:80]  # 61~80
x_test = x[80:]   # 81~100

y_train = y[:60] # 1~60, index from 0 to 59
y_val = y[60:80]  # 61~80
y_test = y[80:]   # 81~100

#2. model
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
for i in range(3):
    model.add(Dense(10))
model.add(Dense(1))

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val)) 

#4. evaluate and predict
results = model.evaluate(x_test, y_test, batch_size=1)
print("mse, mae :", results)

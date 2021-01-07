# input_shape / input_length / input_dim 
# shape = timesteps and features

#1. data
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape) # (4, 3)
print(y.shape) # (4,)

x = x.reshape(4, 3, 1) 

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
# model.add(LSTM(10, activation='relu', input_shape=(3,1))) 
model.add(LSTM(10, activation='relu', input_length=3, input_dim=1)) 
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

# model.summary()

#3. compile and fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

x_pred = np.array([5, 6, 7]) # (3,)
x_pred = x_pred.reshape(1, 3, 1)

#4. evaluate and predict
loss = model.evaluate(x, y)
print(loss)

result = model.predict(x_pred)
print(result)

# 결과 LSTM2
# 0.002442561322823167
# [[7.667309]]
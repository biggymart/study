# keras23_LSTM3.py 카피

import numpy as np

#1. data
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape) # (13, 3)
print(y.shape) # (13,)

x = x.reshape(13, 3, 1)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU

model = Sequential()
model.add(GRU(10, activation='relu', input_shape=(3,1)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=1, callbacks=[early_stopping])

x_pred = np.array([50,60,70]) # (3,)
x_pred = x_pred.reshape(1, 3, 1)

#4. evaluate and predict
loss = model.evaluate(x, y)
print("loss(mse) :", loss)

result = model.predict(x_pred)
print(result)

# 결과 LSTM3
# 1.0227223634719849
# [[80.15487]]

# 결과 GRU2
# 0.6323508620262146
# [[82.68579]]
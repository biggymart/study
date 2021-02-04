# keras23_LSTM1.py 카피

#1. data
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape) # (4, 3)
print(y.shape) # (4,)

x = x.reshape(4, 3, 1) 

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU

model = Sequential()
model.add(GRU(10, activation='relu', input_shape=(3,1))) 
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

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

# 결과 LSTM1
# 0.01851184293627739
# [[8.077794]]

# 결과 GRU1
# 0.0007098008063621819
# [[7.958071]]

# cell state를 빼서 LSTM보다 빠르지만 성능은 비슷함

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# gru (GRU)                    (None, 10)                390
# _________________________________________________________________
# dense (Dense)                (None, 20)                220
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                210
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 11
# =================================================================
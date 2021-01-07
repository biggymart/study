# keras23_LSTM3을 카피
# LSTM 층을 두 개 만드세요
# 에러를 고치고 성능비교

import numpy as np

#1. data
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

# print(x.shape) # (13, 3)
# print(y.shape) # (13,)

x = x.reshape(x.shape[0], x.shape[1], 1)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1), return_sequences=True))
model.add(LSTM(12))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 3, 10)             480
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 12)                1104
# _________________________________________________________________
# dense (Dense)                (None, 20)                260
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                210
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 2,065
# Trainable params: 2,065
# Non-trainable params: 0
# _________________________________________________________________

# 첫번째 LSTM 레이어가 받아들이는 데이터는 3차원 (None, 3, 1), input_dim은 2차원 (3, 1)
# 옵션 없이 default인 경우 LSTM은 3차원인 데이터를 받아서 2차원인 데이터를 출력함 (None, 노드갯수)
# 하지만 return_sequences=True는 받은 데이터의 차원대로 출력하게 함 (None, input_dim, 노드갯수)
# return_sequences=True 옵션을 주었기 때문에 Output Shape이 3차원인 것을 확인할 수 있음
# 두번째 param #: g*(h+i+1)*h, g=4, h=12, i=10 따라서 1104

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

# 결과 keras23_LSTM3
# loss(mse) : 0.10934270173311234
# [[78.521965]]

# ValueError: Input 0 of layer lstm_1 is incompatible with the layer: expected ndim=3, found ndim=2. Full shape received: (None, 10)
'''
SOLUTION
You need to add return_sequences=True to the first layer so that its output tensor has ndim=3 (i.e. batch size, timesteps, hidden state).
# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(10, activation='softmax'))
'''

# 결과 keras28_LSTM
# loss(mse) : 0.1138833612203598
# [[71.82738]]

# 왜 LSTM 두 개 이상을 이으면 성능이 오히려 더 좋지 않을까?
# 첫번째 레이어를 통과한 output은 더 이상 연속적인 데이터가 아니라서 두번째 레이어가 받는 input이 해당 모델과 맞지 않음
# 하지만 두번째 레이어에 주어지는 데이터가 시계열이라 판단이 되면 두 개의 LSTM 레이어 사용 가능
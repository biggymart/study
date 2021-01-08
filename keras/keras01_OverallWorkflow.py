import numpy as np
import tensorflow as tf

#1. 데이터 준비
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델 구성 (The task is hyper-parameter tunning)
from tensorflow.keras.models import Sequential # 신경망에서 한 단계씩 거치니까 순차적인 모델임
from tensorflow.keras.layers import Dense # DNN 을 만들겠다, 가장 기본적인 모델

model = Sequential()
model.add(Dense(5, input_dim=1, activation='linear')) # add : 층을 쌓겠다
model.add(Dense(3, activation='linear')) # 활성화 함수
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
'''
from tensorflow.keras.optimizers import Adam, SGD
optimizer1 = Adam(learning_rate=0.1)
model.compile(loss='mse', optimizer=optimizer1)

optimizer2 = SGD(learning_rate=0.1)
model.compile(loss='mse', optimizer=optimizer2)
'''
model.compile(loss='mse', optimizer='adam') # mean_squared_error
model.fit(x, y, epochs=100, batch_size=1) # 휘트니스센터에서 훈련해라, epochs 훈련하는 회수, batch_size 한칸씩 짤라서 써라

#4. 평가, 예측
loss = model.evaluate(x, y, batch_size=1)
print('Loss :', loss)

result = model.predict([4])
print('Result :', result)

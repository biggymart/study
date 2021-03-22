# 모든 파일을 깃허브에 올릴 것
# loss 이외에 또 다른 지표(metrics)를 올릴 것임
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
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1))

#3. compile and fit
# model.compile(loss='mse', optimizer='adam', metrics=['acc']) # 혹은 accuracy, 0.2라면 20퍼센트만 맞음
# model.compile(loss='mse', optimizer='adam', metrics=['mse']) # loss 값과 동일함
model.compile(loss='mse', optimizer='adam', metrics=['mae']) # 절댓값 먹인 오차값
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. evaluate and predict
loss = model.evaluate(x_test, y_test, batch_size=1) # loss 변수에 대해서 값이 두 개 이상이라서 리스트로 묶임
print('Loss :', loss)

result = model.predict(x_train) 
# accuracy는 0이 나옴, 로스도 낮으니까 잘 된 거 아님? 
# 근데 결과를 보면 완전히 일치하는 값이 아니기 때문에 accuracy는 0임 1과 0.99994는 다르다
# 나중에 라벨이 붙는 자료(개, 고양이 구분 등)에는 좋은 지표지만 숫자는 연속적이기 때문에 accuracy는 사용하지 않음
print('Result :', result)
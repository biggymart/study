import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. data
x = np.array([range(100), range(301, 401), range(1, 101), range(201, 301), range(401, 501)])
x = np.transpose(x)
y = np.array([range(711,811), range(1,101)])
y = np.transpose(y)

x_pred2 = np.array([100, 402, 101, 301, 501])
x_pred2 = x_pred2.reshape(1, 5)

# print(x.shape) # (100,5)
# print(y.shape) # (100,2)
# print(x_pred2.shape) # (1, 5)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. model
model = Sequential()
# model.add(Dense(10, input_dim=5)) # 행무시(가장 첫 번째 인자의 값, 전체 데이터의 갯수), 열우선, input_dim=5 차원이 5개 짜리라는 의미, 스칼라를 사용함
model.add(Dense(10, input_shape=(5,)))
# 스칼라 5개 짜리 벡터 한 개, 즉 우리가 받아들이는 건 column/feature 5개다, 왜 이걸 쓸까? 차원이 2차원 초과하는 것들을 다루기 위해 씀
# (500, 100, 3) 형태의 데이터면 input_shape = (100, 3), input_dim은 안 됨
# 이미지 (가로, 세로, 컬러) (28,28,3)가 10000장이면 (10000,28,28,3)인 4차원 데이터임. 이를 처리하기 위해선 input_shape = (28, 28, 3)
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=1)

#4. evaluate and predict
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss, '\nmae :', mae)

y_predict = model.predict(x_test)
# print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2 :', r2)

y_pred2 = model.predict(x_pred2)
print(y_pred2)
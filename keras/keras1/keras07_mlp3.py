# 다:다 mlp
# 실습
# 1. x는 (100, 5) 데이터 구성; y는 (100, 2) 데이터 구성한 모델을 완성하시오
# 2. predict의 일부값을 출력하시오

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. data
x = np.array([range(100), range(301, 401), range(1, 101), range(201, 301), range(401, 501)]) # 5 features
x = np.transpose(x)
y = np.array([range(711,811), range(1,101)]) # 2 features
y = np.transpose(y)

x_pred2 = np.array([100, 402, 101, 301, 501])
# x_pred2 = np.transpose(x_pred2) # 스칼라로만 이루어진 단벡터이기 때문에 행과 열이 바뀔 여지가 없음
# shape == (5,)인데 1차원, 즉 이 메소드는 적합하지 않음 (행렬은 transpose, 단벡터은 reshape 메소드)
x_pred2 = x_pred2.reshape(1, 5) # 2차원임 (단벡터와 다름) [[100, 402, 101, 301, 501]]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. model
model = Sequential()
model.add(Dense(10, input_dim=5)) # 5 features
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2)) # 2 features

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2) 

#4. evaluate and predict
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss, '\nmae :', mae)

y_predict = model.predict(x_test)
print(y_predict)

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
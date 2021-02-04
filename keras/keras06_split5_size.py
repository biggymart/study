from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. data
x = np.array(range(1, 101))
y = np.array(range(1, 101))

from sklearn.model_selection import train_test_split
# 경우 1: train_size + test_size 값이 1보다 큰 경우
# x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, test_size=0.2, shuffle=False)
# ValueError: The sum of test_size and train_size = 1.1, should be in the (0, 1) range. Reduce test_size and/or train_size.
'''
경우 2: train_size + test_size 값이 1보다 작은 경우
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.2, shuffle=False)
x_train shape : (56,)
y_train shape : (56,)
x_val shape : (14,)
y_val shape : (14,)
x_test shape : (20,)
y_test shape : (20,)
에러는 뜨지 않지만 훈련 데이터와 검정 데이터의 숫자가 줄어들어 모델의 성능이 저하됨
(아래 train_size=0.8과 비교해볼 것)
loss : 8.109458923339844 mae : 2.8340137004852295
RMSE : 2.8477116465484613
R2 : 0.7561064173868346
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=False)
# loss : 0.1361473649740219 mae : 0.364837646484375
# RMSE : 0.3689808094319125
# R2 : 0.9959053582637886

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
print('x_train shape :', x_train.shape) # (64,)
print('y_train shape :', y_train.shape) # (64,)
print('x_val shape :', x_val.shape) # (16,)
print('y_val shape :', y_val.shape) # (16,)
print('x_test shape :', x_test.shape) # (20,)
print('y_test shape :', y_test.shape) # (20,)

#2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

#4. evaluate and predict
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss, 'mae :', mae)

y_predict = model.predict(x_test)
print('y_predict :', y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 :", r2)

# 요약
# 1. sklearn.model_selection의 train_test_split 옵션 중 train_size와 test_size의 합이 1이 되도록 해야 함
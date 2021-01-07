from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. data
x = np.array(range(1, 101))
y = np.array(range(1, 101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
# 실습: validation_data 를 만들 것
# (조건: train_test_split 활용; 데이터 비율 train : val : test = 64 : 16 : 20)
print('x_train shape :', x_train.shape)
print('y_train shape :', y_train.shape)
print('x_val shape :', x_val.shape)
print('y_val shape :', y_val.shape)
print('x_test shape :', x_test.shape)
print('y_test shape :', y_test.shape)

#2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
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
# 1. sklearn.model_selection의 train_test_split을 활용하여 validation data를 만들고
# 2. fit의 옵션으로 validation_data=(x_variable, y_variable)
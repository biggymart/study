# 실습
# R2를 음수가 아닌 0.5 이하로 줄이기 (모델을 개판으로 만들어라)
# 1. 레이어는 인풋과 아웃풋을 포함 5개 이상
# 2. batch_size = 1
# 3. epoches = 100 이상
# 4. 히든레이어의 노드의 갯수는 10 이상
# 5. 데이터 조작 금지.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


#1. data
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15])
y_test = np.array([11, 12, 13, 14, 15])
x_pred = np.array([16, 17, 18])

#2. model
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
for i in range(50):
    model.add(Dense(15))
model.add(Dense(1))

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2) 

#4. evaluate and predict
results = model.evaluate(x_test, y_test, batch_size=1)
print("mse, mae :", results)

y_predict = model.predict(x_test)
# print("y_predict :", y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 :", r2)

# 결과 keras06_R2
# mse, mae : [0.40966886281967163, 0.6288830041885376]
# RMSE : 0.640053457273806
# R2 : 0.795165785915924

# 결과 keras07_R2_test
# mse, mae : [0.1463325321674347, 0.37827739119529724]
# RMSE : 0.38253453442212154
# R2 : 0.9268336649872253
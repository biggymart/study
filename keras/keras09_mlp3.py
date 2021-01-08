# 다:다 mlp
# x1, x2, x3 = 환율, 금리, 국제 유가; y1, y2, y3 = 삼성전자주가, 내일환율, 내일금리;
# 다음의 수식과 같이 모델링 됨: w1x1 + w2x2 + w3x3 + b = y1 + y2 + y3

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. data
x = np.array([range(100), range(301, 401), range(1, 101)])
x = np.transpose(x)
y = np.array([range(711,811), range(1,101), range(100)])
y = np.transpose(y)
# shape == (100, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. model
model = Sequential()
model.add(Dense(10, input_dim=3)) # feature 3개
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3)) # output 3개

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100 ,batch_size=1, validation_split=0.2)

#4. evaluate and predict
loss, mae = model.evaluate(x_test, y_test)
print( 'loss :', loss, "\nmae :", mae)

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

# 결과
# loss : 1.580464314976382e-09
# mae : 2.9283266485435888e-05
# RMSE : 3.975505460676403e-05
# R2 : 0.9999999999980003
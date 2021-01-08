# 실습
# 1. R2         : 0.5 이하 / 음수 불가
# 2. layer      : 5개 이상
# 3. node       : 각 10개 이상
# 4. batch_size : 8 이하
# 5. epochs     : 30 이상

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
x_pred2 = x_pred2.reshape(1, 5)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. model
model = Sequential()
model.add(Dense(10, input_dim=5)) # 5 features
for i in range(130):
    model.add(Dense(30))
model.add(Dense(2)) # 2 features

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=30, batch_size=8, validation_split=0.2) 

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

# 9 layers, 10 nodes
# RMSE : 20.364197717015927
# R2 : 0.4752801229139606

# model = Sequential()
# model.add(Dense(10, input_dim=5)) # 5 features
# # for i in range(120):
#     model.add(Dense(30))
# model.add(Dense(2))
# RMSE : 19.361287391902327
# R2 : 0.5256909958569883
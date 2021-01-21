# keras18_boston3_MinMaxScaler.py 카피

#1. data
import numpy as np
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# 평균을 중심으로 양쪽으로 분산하게 만들어서 데이터를 가운데로 모아주는 효과
# 평균 자체가 중심이 한쪽으로 쏠려있기 때문에 완전히 중심으로 가져오진 못함
# 이상치의 효과를 약화시키고 중심값 위주로 모아줌

# MinMaxScaler
# print(np.max(x), np.min(x)) # 711.0 0.0 => 1.0 0.0
# print(np.max(x[0]))         # 0.44105193260704206

# StandardScaler
print(np.max(x), np.min(x)) # 711.0 0.0 => 9.933930601860268 -3.9071933049810337
print(np.max(x[0]))


# 실습: 모델을 구성하시오
#2. model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
input1 = Input(shape=(13,))
dense1 = Dense(128, activation='relu')(input1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1, outputs=output1)

#3. compile and fit
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, batch_size=4, epochs=150, verbose=1, validation_split=0.2)

#4. evalutate and predict
mse, mae = model.evaluate(x_test, y_test, batch_size=4)
print("mse :", mse, "\nmae :", mae)

y_predict = model.predict(x_test)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2 :', r2)

# MinMaxScaler
# mse : 7.581524848937988 
# mae : 2.1726179122924805
# RMSE : 2.7534571507275873
# R2 : 0.9232851710250565

# StandardScaler
# mse : 8.463397979736328
# mae : 2.2769787311553955
# RMSE : 2.909192101656423
# R2 : 0.9143618107277143
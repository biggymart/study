# Scaling (MinMaxScalar) : validation_data까지 한 버전
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

#1. data
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
_, x_val, _, y_val = train_test_split(x, y, test_size=0.2, random_state=1) # 전처리

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val) # 전처리

# 실습: 모델을 구성하시오
#2. model
input1 = Input(shape=(13,))
dense1 = Dense(128, activation='relu')(input1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1, outputs=output1)

#3. compile and fit
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, batch_size=4, epochs=150, verbose=1, validation_data=(x_val, y_val))
# split 대신 data 넣어줌

#4. evalutate and predict
mse, mae = model.evaluate(x_test, y_test, batch_size=4)
print("mse :", mse, "\nmae :", mae)

y_predict = model.predict(x_test) # 단순히 지표를 만들기 위해 만든 변수임

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
# mse : 8.84180736541748
# mae : 2.3092715740203857
# RMSE : 2.9735179485902052
# R2 : 0.9105328083803463
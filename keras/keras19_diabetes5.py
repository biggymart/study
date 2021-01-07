# 실습: keras18을 참고하여 총 6가지의 버전을 만드시오
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

#1. data
dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
_, x_val, _, y_val = train_test_split(x, y, test_size=0.2, random_state=1) # 전처리

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
x_val = scaler.transform(x_val) # 전처리

#2. model
input1 = Input(shape=(10,))
dense1 = Dense(64, activation='relu')(input1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1, outputs=output1)

#3. compile and fit
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, batch_size=4, epochs=100, verbose=1, validation_data=(x_val, y_val))

#4. evaluate and predict
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

# 결과값 ver1
# mse : 3719.33251953125
# mae : 47.651283264160156
# RMSE : 60.986330591424206
# R2 : 0.3942092465679128

# 결과값 ver2
# mse : 3216.4775390625 
# mae : 45.81156921386719
# RMSE : 56.71399227602712
# R2 : 0.45881102256996287

# 결과값 ver3
# mse : 3523.6650390625
# mae : 47.037574768066406
# RMSE : 59.36046900662257
# R2 : 0.44497624021969806

# 결과값 ver4
# mse : 3020.0166015625 
# mae : 43.657318115234375
# RMSE : 54.95467608233062
# R2 : 0.476487012334076

# 결과값 ver5
# mse : 3412.5498046875
# mae : 46.411434173583984
# RMSE : 58.41703813079293
# R2 : 0.3988953866834143

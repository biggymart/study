# scikitlearn, LSTM으로 모델링하고 Dense와 성능 비교하라 (회귀)

# keras19_diabetes6.py 카피
#1. data
import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

#1-1. preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
_, x_val, _, y_val = train_test_split(x, y, test_size=0.2, random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
x_val = scaler.transform(x_val)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1) # LSTM용 데이터로 가공
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, LSTM
input1 = Input(shape=(10,1))
lstm1 = LSTM(200, activation='relu')(input1)
dense1 = Dense(200, activation='relu')(lstm1)
dense1 = Dense(200, activation='relu')(dense1)
dense1 = Dense(200, activation='relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1, outputs=output1)

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, batch_size=1, epochs=1000, validation_data=(x_val, y_val), callbacks=[early_stopping])

#4. evaluate and predict
mse, mae = model.evaluate(x_test, y_test, batch_size=1)
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

# 결과값 keras19_diabetes6.py
# mse : 3199.4365234375
# mae : 43.93169403076172
# RMSE : 56.56355713659885
# R2 : 0.5236264001339381

# 결과값 keras33_LSTM2_diabetes.py
# mse : 5488.6533203125
# mae : 52.69452667236328
# RMSE : 74.0854482472466   
# R2 : -0.029965133978495873
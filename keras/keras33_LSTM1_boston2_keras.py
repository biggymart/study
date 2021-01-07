# keras
# LSTM으로 모델링하고
# Dense와 성능 비교하라
# 회귀

# keras20_boston_keras2.py 카피
#1. data
import numpy as np
from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.2, seed=113)

#1-1. preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1) # LSTM용 데이터로 가공
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, LSTM

input1 = Input(shape=(13, 1))
lstm1 = LSTM(128, activation='relu')(input1)
dense1 = Dense(128, activation='relu')(lstm1)
dense1 = Dense(128, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1, outputs=output1)

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=15, mode='min')

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, batch_size=1, epochs=150, validation_split=0.2, callbacks=[early_stopping])

#4. evaluate and predict
mse, mae = model.evaluate(x_test, y_test, batch_size=1)
print("mse :", mse, "\nmae :", mae, sep='')

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

# 결과 keras20_boston_keras2.py
# mse : 12.786617279052734
# mae : 2.4296789169311523
# RMSE : 3.5758383028528646
# R2 : 0.8463956220728037

# 결과 keras33_LSTM1_boston2_keras.py
# mse :22.557918548583984
# mae :3.0959410667419434
# RMSE : 4.749517975701783
# R2 : 0.729013958356949
# scikitlearn, LSTM으로 모델링하고 Dense와 성능 비교하라 (회귀)
# keras18_boston5_MinMax_val.py 카피

#1. data
import numpy as np
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target

# import numpy as np
# from tensorflow.keras.datasets import boston_housing
# (x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.2, seed=113)

#1-1. preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
_, x_val, _, y_val = train_test_split(x, y, test_size=0.2)

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
input1 = Input(shape=(13, 1))
lstm1 = LSTM(64, activation='relu')(input1)
dense1 = Dense(64, activation='relu')(lstm1)
dense1 = Dense(32, activation='relu')(dense1)
dense1 = Dense(32, activation='relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1, outputs=output1)

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, batch_size=4, epochs=1000, validation_data=(x_val, y_val), callbacks=[early_stopping])

#4. evalutate and predict
mse, mae = model.evaluate(x_test, y_test, batch_size=4)
print("mse :", mse, "\nmae :", mae, sep='')

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

# 결과 keras18_boston5_MinMax_val
# mse : 8.84180736541748
# mae : 2.3092715740203857
# RMSE : 2.9735179485902052
# R2 : 0.9105328083803463

# 결과 keras33_LSTM1_boston1_sklearn
# mse :23.26708984375
# mae :3.195652484893799
# RMSE : 4.823597737472614
# R2 : 0.7142568845429783
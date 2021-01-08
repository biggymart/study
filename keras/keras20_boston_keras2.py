# 실습> 2개의 파일을 만드시오:
# 1. EarlyStopping을 적용하지 않은 최고의 모델
# 2. EarlyStopping을 적용한 최고의 모델
# (단, 사용 모듈: from tensorflow.keras.datasets import boston_housing)
# Tip: sklearn과 제공한 데이터와 비슷하지만 x와 y로 나누지 않음

#1. data
import numpy as np
from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data(test_split=0.2, seed=113)
# load_data() 함수를 통해 훈련 데이터와 테스트 데이터로 나누게 된다

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) # 전처리

#2. model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

input1 = Input(shape=(13,))
dense1 = Dense(128, activation='relu')(input1)
dense1 = Dense(128, activation='relu')(dense1)
dense1 = Dense(128, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1, outputs=output1)

#3. compile and fit
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=15, mode='min')

model.fit(x_train, y_train, batch_size=1, epochs=150, verbose=1, validation_split=0.2, callbacks=[early_stopping],)

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

# 결과 ver 1
# mse : 16.36920928955078
# mae : 2.69653582572937
# RMSE : 4.045888636971458
# R2 : 0.8033582642580341

# 결과 ver 2
# mse : 12.786617279052734
# mae : 2.4296789169311523
# RMSE : 3.5758383028528646
# R2 : 0.8463956220728037
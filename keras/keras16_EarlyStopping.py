# Early Stopping : 
# loss가 처음에는 줄어들다가 어느 시점이 지나가면 성능이 정체되고 오히려 떨어진다 (과적합 구간 진입)
# 그 시점 이전에 멈춰야 한다 (epoch 조절), epoch는 model.fit에 있으니까 거기서 뭔가 하는거임
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

#1. data
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
_, x_val, _, y_val = train_test_split(x, y, test_size=0.2) # 전처리

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

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto') 
# loss 로 모니터링 기준을 삼고, 
# 내가 원하는 최저 혹은 최고 값을 찍고 n번 동안은 참고 봐줄게, n번 이내에 새로운 기록 찾아내!
# 나중에 보강할 부분이 있을 것
# 돌려보니 epoch=262에서 끝났다! 그럼 epoch=242로 하면 다 해결될까?
# 그런데 다시 돌리면 새로운 세트로 만들어져서 w가 달라짐. 그래서 나중에는 w값을 남기는 방식을 배울 것임.

model.fit(x_train, y_train, batch_size=4, epochs=2000, verbose=1, validation_data=(x_val, y_val), callbacks=[early_stopping])

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

# 결과 keras18_EarlyStopping
# Epoch 214/2000
# mse : 8.512691497802734
# mae : 1.930296778678894
# RMSE : 2.917651639027537
# R2 : 0.90123263762414
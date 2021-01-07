# Scaling (MinMaxScalar) : validation_data까지 전처리함
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

# 데이터 전처리 (minMaxScalar): 통상적으로 성능이 향상됨 **전처리는 반드시 해야함**
from sklearn.preprocessing import MinMaxScaler # 정규화 normalization
scaler = MinMaxScaler() # 각 열별로 똑똑하게 최솟값 최댓값 적용해서 계산해줌, 편-안
scaler.fit(x_train) # 가중치 계산해줌, **x_train을 기준**으로 전처리를 해줘야 과적합의 오류를 피할 수 있음
x_train = scaler.transform(x_train) # 가중치에 맞게 x_train 값을 0 - 1 범위로 변환해줌 
x_test = scaler.transform(x_test) # 미리 계산한 가중치에 맞게 x_test 값을 변환해준다. 
# target 값(y)은 건들지 마라

'''
전처리   X              Y
train   fit, transform 세모(존재하지만, 전처리하지 않는다)
test    transform      세모
val     transform      세모
predict transform      엑스(존재하지 않음) 
'''

# print(np.max(x), np.min(x)) # 1.0 0.0

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

# 결과
# mse : 9.331316947937012
# mae : 2.31037974357605
# RMSE : 3.0547202311546857
# R2 : 0.905579660243128
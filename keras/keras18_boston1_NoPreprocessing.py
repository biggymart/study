# 실질적인 데이터로 맨땅부터 모델만들기
# 여태까지는 데이터가 실질적이지 않은 값으로 실습했다
# 이번에는 정말 실제로 있는 데이터로 해보도록 하자 -> 보스턴 집값 (boston housing)
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html

'''
# 데이터 파악
print(x.shape, y.shape) # (506, 13) (506,) -> feature 13개, 아웃풋은 1개구나
print(x[:5]) 
# 숫자가 6.3200e-03 같은 형식으로 나오는데, 숫자가 너무 커져서 터지는 걸 방지함
# 데이터 전처리가 이렇게 중요하다, 생데이터를 만나면 이런 식으로 변환해야 함
print(y[:5])
# 숫자가 21.6처럼 나오는데 값이 '이냐 아니냐(0 혹은 1; 분류)'가 아니라서 회귀모델이다

print(np.max(x), np.min(x))
# 711.0 0.0 처럼 나오는데, 데이터 정규화가 필요함, 참고로 전처리방식은 6가지 있음
# 이상치도 제어해야 함, 이상치라는 근거는? 모델을 이상하게 만드는 것 e.g. 직장인 평균연봉에 삼성전자 부회장 이재용 포함
print(dataset.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']
print(dataset.DESCR)
# 자료에 대한 설명이 나옴, 교육용 데이터라서 있는 것이지 일반적으로 없음
'''

# 실습: 모델을 구성하시오

#1. data
import numpy as np
from sklearn.datasets import load_boston # 교육훈련용

dataset = load_boston() # sklearn에서 제공하는 load_boston 자료는 이렇게 언팩킹함 (keras와 상이함)
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

#2. model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

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

# 결과 ver1
# mse : 20.25518035888672
# mae : 3.4258222579956055
# RMSE : 4.500575586257058
# R2 : 0.7950448684606252
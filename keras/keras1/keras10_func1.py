import numpy as np
from tensorflow.keras.models import Sequential, Model # ctrl + space (대문자는 주로 class형), 함수형 모델
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split


#1. data

x = np.array([range(100), range(301, 401), range(1, 101), range(201, 301), range(401, 501)])
x = np.transpose(x)
y = np.array([range(711,811), range(1,101)])
y = np.transpose(y)

x_pred2 = np.array([100, 402, 101, 301, 501])
x_pred2 = x_pred2.reshape(1, 5)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# #2. model

#  시퀀셜 모델이 아니고 함수형 모델임 (재사용 용이)
input1 = Input(shape=(5,)) # 인풋레이어를 직접 구성하겠다
dense1 = Dense(5, activation='relu')(input1) # 시퀀셜 모델과 다른 문법임, 전 레이어의 아웃풋은 현 레이어의 인풋
dense2 = Dense(3)(dense1)
dense3 = Dense(4)(dense2)
outputs = Dense(2)(dense3)
model = Model(inputs=input1, outputs=outputs) # 처음과 끝을 명시해줌, 모델을 다 구성한 후 형을 선언한다
# model.summary() # 레이어 노드: 1, 5, 3, 4, 1, 파라미터 갯수: 49

'''
# 위 모델과 표현방식만 다를 뿐 성능이 똑같다
model = Sequential() # 모델을 구성하기 전 형을 선언한다
# model.add(Dense(10, input_dim=1))
model.add(Dense(5, activation='relu', input_shape=(5,)))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(2))
# model.summary() # 레이어 노드: 5, 3, 4, 1, 파라미터 갯수: 49
'''

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=1)

#4. evaluate and predict
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss, '\nmae :', mae)

y_predict = model.predict(x_test)
# print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2 :', r2)

y_pred2 = model.predict(x_pred2)
print(y_pred2)

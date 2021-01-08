# 앙상블 (ensemble): 모델 2개 이상 사용
# 예를 들어 주식에서 삼성전자, 하이닉스, 하이트 주가를 바탕으로 내일 삼성전자 주가를 예측한 모델과
# 또, 온도 습도 미세먼지 농도로 강수량을 예측한 모델을 만들었다. 
# 어느 날, 두 모델을 합쳐서 강수량을 예측해보면 어떨까?
# 그러면 한 모델의 가중치는 상대방 모델의 가중치에 영향을 미치게 된다. 

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate, LSTM, Conv2D
# 병합을 위해 concatenate 임포트, 아래는 옛날 버전
# from keras.layers.merge import concatenate, Concatenate
# from keras.layers import concatenate, Concatenate


#1. data
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x1 = np.transpose(x1)
y1 = np.array([range(711,811), range(1,101), range(201, 301)])
y1 = np.transpose(y1)

x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
x2 = np.transpose(x2)
y2 = np.array([range(501, 601), range(711, 811), range(100)])
y2 = np.transpose(y2)

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle=False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, shuffle=False)

#2. model
# 현재 수준에서는 함수형 모델로 만드는 게 적합함

# 모델 1
input1 = Input(shape=(3,), name='model1_input')
dense1 = Dense(11, activation='relu', name='model1_layer1')(input1)
dense1 = Dense(6, activation='relu', name='model1_layer2')(dense1)
# output1 = Dense(3)(dense1)

# 모델 2
input2 = Input(shape=(3,), name='model2_input')
dense2 = Dense(10, activation='relu', name='model2_layer1')(input2)
dense2 = Dense(4, activation='relu', name='model2_layer2')(dense2)
dense2 = Dense(3, activation='relu', name='model2_layer3')(dense2)
# output2 = Dense(3)(dense2)

# Google: 모델 병합 / concatenate (사전적 정의, tensorflow 공식 사이트): 사슬로 엮다
merge1 = concatenate([dense1, dense2], name='cat_start') # 이것도 하나의 layer임
middle1 = Dense(15, name='cat_layer1')(merge1) # 이어서 layer 만들기
middle1 = Dense(10, name='cat_layer2')(middle1)
middle1 = Dense(12, name='cat_layer3')(middle1)
# 여기에도 (그냥 분기했을 수도 있겠지만) 레이어를 쌓았기 때문에 하나의 모델이라고 볼 수 있음

# 모델 분기 1
output1 = Dense(30, name='out1_layer1')(middle1)
output1 = Dense(7, name='out1_layer2')(output1)
output1 = Dense(3, name='out1_layer3')(output1)

# 모델 분기 2
output2 = Dense(15, name='out2_layer1')(middle1)
output2 = Dense(7, name='out2_layer2')(output2)
output2 = Dense(9, name='out2_layer3')(output2)
output2 = Dense(3, name='out2_layer4')(output2)

# 여기까지 5개의 모델이 있다고 할 수 있음

# 모델 선언
model = Model(inputs=[input1, input2], outputs=[output1, output2]) # 두 개 이상은 리스트
# model.summary()

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=10, batch_size=1, validation_split=0.2, verbose=1)

#4. evaluate and predict
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print( 'loss :', loss)
# 값이 다섯개 나오는데 
# loss : [1111.803466796875, 670.62451171875, 441.178955078125, 21.69460678100586, 16.086742401123047]
# 2번째, 3번째 합치니까 1번째
# [전체 모델의 loss 합 혹은 대표 loss, 첫번째 모델의 loss, 두번째 모델의 loss, 첫번째 모델의 metrics, 두번째 모델의 metrics]
print("model.metrics_names :", model.metrics_names)
# model.metrics_names : ['loss', 'dense_10_loss', 'dense_14_loss', 'dense_10_mae', 'dense_14_mae']


y1_predict, y2_predict = model.predict([x1_test, x2_test])
print("y1_predict :\n", y1_predict) # shape (20,3)의 결과가 출력됨
print("y2_predict :\n", y2_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', (RMSE(y1_test, y1_predict) + RMSE(y2_test, y2_predict))/2)
# 한 숫자로 전체 모델을 대표하게 하는 통계값으로 평균을 쓸 수 있다

# R2 구하기
from sklearn.metrics import r2_score
print('R2 :', (r2_score(y1_test, y1_predict) + r2_score(y2_test, y2_predict))/2)
# RMSE와 마찬가지로 R2도 평균값을 출력하면 된다
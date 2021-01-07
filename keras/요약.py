# 윤영선, 딥러닝으로 걷는 시계열 예측

'''
=== introduction ===
keras00_numpy: numpy 라이브러리 import와 np.array 만들 수 있다, aliss의 개념도 덤으로 알게된다
keras01_OverallWorkflow: 전반적인 작업순서를 파악할 수 있다 "data -> model -> compile and fit -> evaluate and predict", 모델은 Sequential, 레이어는 Dense

=== data ===
keras02_1: 데이터를 훈련과 평가 세트로 나눌 필요성을 알게된다, 모델의 활성화 함수 'relu' 등장
keras02_2: 예측의 개념과 predict 데이터를 만들 필요성을 알게된다

=== model evaluation ===
keras03_metrics: compile에 metrics가 등장, mse와 mae의 개념을 따로 학습
keras04_validation: fit에 validation_split 옵션이 등장, validation을 해야할 필요성을 알게된다
keras05_RMSE: RMSE 개념을 따로 학습하고, scikitlearn과 함수의 개념 등장
keras06_R2: R2 개념 등장 (회귀모델 지표)
keras07_R2_test: hyperparameter tuning을 통해 R2 지표를 끌어올리는 실습

=== data split ===
keras08_split1: list slicing
keras08_split2: from sklearn.model_selection import train_test_split
keras08_split3_val: model.fit(validation_split=0.2)
keras08_split4_val2: model.fit(validation_data=(x_val, y_val))
keras08_split5_size: train_test_split(train_size=float, test_size=float), a + b == 1

'''

# 기초특강 4일 요약
# 수학: 선형회귀 y = wx + b
# 모델 만드는 순서: (1) data, (2) model, (3) compile and fit, (4) evaluate and predict
# 사람의 역할: 정제된 데이터 제공 (train, validation, test, predict)
# 기계의 역할: optimizer를 통해 최소의 loss를 만족하는 최적의 weight and bias 계산

# 08split 요약 모음
# keras08_split1.py
# 1. 리스트 슬라이싱을 활용한 data split
# 2. sklearn.metrics의 mean_squeared_error, r2_score

# keras08_split2.py
# 1. sklearn.model_selection의 train_test_split 용법
# 2. shuffle 옵션의 중요성

# keras08_split3_val
# 1. model.fit에 validation_split=flaot_value 활용하여 data를 split 하는 방식

# keras08_split4_val2
# 1. sklearn.model_selection의 train_test_split을 활용하여 validation data를 만들고
# 2. fit의 옵션으로 validation_data=(x_variable, y_variable)

# keras08_split5_size
# 1. sklearn.model_selection의 train_test_split 옵션 중 train_size와 test_size의 합이 1이 되도록 해야 함

# next lesson...
# 지금까지는 y = wx + b 여서 하나의 x만 입력했다. 
# 하지만 실제 문제를 한 번 생각해보자. 날씨를 알아보려고 할 때 우린 여러 요소를 알아야 한다. 
# 예를 들어 온도, 습도, 강수량 등을 알아야 하는데, 이것은 기존의 y = wx + b에 다 들어갈 수 없다. 
# 이제는 y = w1x1 + w2x2 + w3x3 + b 같은 다항식으로 확장할 필요가 있다. 즉 MLP

# ensemble과 summary

#0106 review

# 어제는 RNN을 했다. 주로 시계열 데이터에서 쓰임
# SimpleRNN, GRU, LSTM
# 게이트 4개, activation='tanh' (default) but we used 'relu'
# summary, param #, g * (i + h + b) * h
# 원래 데이터가 (10,4)이었는데 LSTM을 하기 위해 (10, 4, 1)로 reshape
# LSTM(10, activation='relu', input_shape=(4,1))
# DNN도 3차원 데이터를 받아들이지만 output이 달랐다, 그래서 통상 2차원을 대상으로 쓴다고 하자
'''
    데이터구조      input_shape 
DNN (행, 열)        =(열,)
RNN (행, 열, 몇)    =(열, 몇)
CNN (행, h, w, f)   =(h, w, f)
'''
# SimpleRNN의 문제: 훈련 효과가 초반에 있는 데이터에 반영이 안 됨
# LSTM의 문제: 역전파 때문에 연산이 너무 많아져서 느려짐
# 역전파

#0107 review
# 어제 역전파 개념했었고, LSTM의 되돌림, y = wx + b 나오면 퍼셉트론 떠올려야 하고
# Output Shape은 ruturn sequences=True에 영향을 받음, Input shape을 그대로 output에 전달
'''
예를 들어, 다음이라고 해보자
LSTM(15, input_shape=(5,1), return_sequences=?)
LSTM(4)
입력되는 데이터 shape은 (None, 5, 1), return_sequences가
True인 경우, (None, 5, 1) -> (None, 5, 15) -> (None, 4)  (첫번째 레이어에서 데이터의 차원이 유지됨)
False인 경우, (None, 5, 1) -> (None, 15) -> 에러
'''
# 앙상블
# Model1: LSTM, Model2: LSTM Dense 이렇게 다른 모델 가능한 걸 확인함

# row, column
# (100, 3) (200, 3) 행이 다르므로 불가능
# (100, 2) (100, 4) 열이 달라도 가능

# 결측치 (데이터가 비는 것)
# 중간에 데이터가 비면 (1) 그 행을 제외하거나 (2) 예측치를 사용함

# 시계열 데이터가 주어졌을 때 split 함수를 활용하여 y를 만들 수 있다
# 또한 np.array slicing  --> array[row, column]

# 경사하강법(gradient_descent), **learning_rate** (w 만큼 중요함)
# "우리는 이미 배웠다", 최적의 w는 loss (혹은 cost)가 가장 작은 지점
# 코드에서는 어떻게 반영되어있나 --> keras01.py 참고, optimizer로 learning_rate 설정 (default: 0.01)
'''
from tensorflow.keras.optimizers import Adam, SGD
optimizer1 = Adam(learning_rate=0.1)
model.compile(loss='mse', optimizer=optimizer1)

optimizer2 = SGD(learning_rate=0.1)
model.compile(loss='mse', optimizer=optimizer2)
'''
# SGD (Stochastic Gradient Descent): 가장 기본적인 optimizer
# Gradient vanishing (학습률 너무 낮음, global loss가 아닌 local loss에 빠져버리는 현상), Gradient explosion (학습률 너무 높음)

# LSTM 시간 너무 걸림
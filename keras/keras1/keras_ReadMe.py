# 윤영선, 딥러닝으로 걷는 시계열 예측, 93학번
# 네이버블로그 gema0000

'''
=== introduction ===
00_numpy: import numpy as np; np.array()
01_OverallWorkflow: 전반적인 작업순서 "data -> model -> compile and fit -> evaluate and predict", 모델은 Sequential, 레이어는 Dense

=== data ===
02: 데이터를 훈련과 평가 세트로 나눌 필요성을 알게된다, 모델의 활성화 함수 'relu' 등장, 예측의 개념과 predict 데이터를 만들 필요성을 알게된다

=== model evaluation ===
03_metrics: compile(metrics=['mse','mae']); mse와 mae 개념
04_validation: fit(validation_split=float); validation의 필요성
05_RMSE: RMSE, R2 개념, scikit-learn과 함수의 개념 등장

=== data split ===
08_split1: list slicing
08_split2: from sklearn.model_selection import train_test_split
08_split3_val: model.fit(validation_split=0.2)
08_split4_val2: model.fit(validation_data=(x_val, y_val))
08_split5_size: train_test_split(train_size=0.8, test_size=0.2), a + b == 1

=== MLP (Multi-Layered Perceptron, DNN) ===
09_mlp: 다:1, 다:다

=== more parameters ===
10_verbose: fit(verbose=[0-3])
11_input_shape: Dense(input_shape=(dim,))

=== model construction ===
12_func: 함수형 모델
14_ensemble: ensemble 개념 소개, 2:2, 2:1, 2:1, 2:3, 2:2, 1:2
16_Concatenate: # Concatenate()([mod1, mod2]) # concatenate([mod1, mod2])
17_summary: 모델 만든 후 확인용
18_boston: MinMaxScaler 전처리, sklearn 자료
18_EarlyStopping: fit 제어도구, 

19_diabetes: sklearn 자료 실습
20_boston_keras: keras 자료
21_cancer: sklearn 자료, binary classification
22_iris_1: sklearn 자료, categorical classification,tensorflow.keras.utils.to_categorical, softmax
22_iris_2: sklearn.preprocessing.OneHotEncoder
22_wine: sklearn 자료

=== RNN ===
23_LSTM: 3차원 데이터, shape의 중요성
24_SimpleRNN
25_GRU
28_LSTM_return_sequences: layer parameter 중 하나 
30_LSTM_ensemble: 행 맞춰야 한다
31_split_timeseries: 일반적인 array에서 시계열 데이터로 만드는 함수

=== save and load model ===
35_save_model: model.save('path')


history
dropout
CNN
MCP
save and load npy, weight, MCP
pandas
Conv1D

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

######
# 0108
# Keywords: overfit, dropout, CNN, shape, hist and plt

# 과적합: 어떠한 훈련 데이터에 대해서 모델을 완벽하게 맞게 해서 만들면 나중에 다른 데이터가 들어오면 쓸모가 없음
# test/validation에서 조금만 달라져도 정확도가 낮아짐, 훈련할 때 버릴 애들은 버리는 게 낫다
# 그렇다면 과적합이 되지 않게 하려면 어떻게 해야 할까
# 1. 훈련 데이터를 늘린다
# 2. feature를 줄인다
# (feature가 많다는 건 y = w1x1 + w2x2 + w3x3 + ... 이렇게 연산이 많아짐
# 버릴 특성은 버린다, mnist할 때 600여개 feature이 있는데 100개 정도로 줄여야 한다)
# 3. regularization (정규화) 실시
# 4. Dropout (DL 됨, ML 안 됨)
# LSTM 시간 너무 걸림

# DNN => Regression,    (row, col)
# RNN => Time-series,   (row, col, 1)
# CNN => Image,         (N, row, col, rgb)


######
# 0111
# Keywords: plt, ModelCheckPoint

# 6x6 자료가 있다고 하면
# (36, 1), (18, 2), (12, 3), (9, 4), (6, 6), (4, 9), (3, 12), (2, 18), (1, 36)
# 이러한 조합이 가능한데 그 중 (9, 4), (4, 9) 같은 경우에는 자료가 원래 형태를 유지하지 않고
# 짤리게 된다. 한 줄에 6개 인데 4개씩 자르면 2번째 줄 앞 2개가 위에 있는 것과 함께 묶이게 된다.
# 짜를 때 약수로 묶는 게 좋다.

# Early Stopping의 문제점
# 마지막 최저점을 1 patience cycle만큼 지나쳐서 stop하게 된다.
# 매 최저점의 지점을 찍히는 코드를 만들면 어떨까

# 0112
# Keywords: load_model, save_weights, load_weights
# numpy data 저장하고 불러오기 
# 모델 저장하고 불러오기 (지점이 중요함: #2 이후 혹은 #3 이후)
# <가중치 저장>이란 개념이 처음 등장! => 또 다시 훈련할 필요가 없어서 시간을 많이 절약할 수 있음

# fit: 공략법을 얻기 위해 주구장창 게임을 play 하는 것
# weight: 여러 시행착오를 통해 얻은 공략법

# 0113
# numpy를 하다보면 느끼겠지만 한 종류의 자료형만 쓰는 것을 알았을 거야
# R에서 가져온 것: pandas 자료형에 대해서 flexible 함; header, index 기본적으로 제공
# pandas에서 가장 기본이 되는 자료:
#   dataframe: Two-dimensional, size-mutable, potentially heterogeneous tabular data.
#   series:    One-dimensional ndarray with axis labels (including time series).

# pandas로 데이터프레임을 만들 수 있고, csv로 저장하고 불러올 수 있다

# 0120
# https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/Underfit model: train_loss < val_loss
# LSTM basics
# underfit model: train_loss < val_loss
# (1) can be improved by increasing the number of the training epochs.
# (2) underprovisioned (insufficient memory cells): increase the number of memory cells in a hidden layer or number of hidden layers.

# Quantile Regression
# point estimate: the estimate that minimizes squared deviations from reality
# prediction interval: likely range rather than a single estimate
# the most popular quantile is the median (50th percentile)
# The quantile loss differs depending on the evaluated quantile, such that more negative errors are penalized more for higher quantiles and more positive errors are qenalized for lower quantile.

# 0121
# Keyword: StandardScaler
# 표준점수로 변환 -> 표준편차
# 편차: 관측값 - 중심값(평균 혹은 중앙값)
# 분산: 편차의 제곱의 평균
# 표준편차: 분산의 제곱근
# 정규분포: 중심이 0이고 표준편차가 1인 분포
# Standard Scaler: 한쪽으로 치우친 것을 좀 고르게 만들어줌

# 0125
## 데이터 이해하기 (간단한 EDA)

# 1. 데이터의 사이즈는? 모델 학습에 적합한 형태인가?
# 2. Train/Test는 어떻게 분리되어 있는가?
# 3. Missing Value는?
# 4. Target Variable의 분포는?
# 5. 간단히 데이터 살펴보기
# 6. 데이터의 특이한/주목해야할 부분은?

# 0127
# 1교시: keras45 datetime 추가하는 방법, string에 대한 join() ## 과제 실시간 업데이트되게 바꿔라
# 2교시: keras01_2 scatter, difference btwn DL and ML (you mei you hidden layer)
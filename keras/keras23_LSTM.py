# 연속된 데이터
# 주의해야 하는 부분: X의 shape, input의 shape

# input_shape / input_length / input_dim 
# shape = timesteps and features

#1. data
import numpy as np

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = np.array([4,5,6,7])

print(x.shape) # (4, 3)
print(y.shape) # (4,)

x = x.reshape(4, 3, 1) 
# shape은 바꿨지만 요소는 같음;
# [[[1], [2], [3]],
#  [[2], [3], [4]],
#  [[3], [4], [5]],
#  [[4], [5], [6]]]
# (당분간은) 하나씩 잘라서 연산해서 shape을 바꿔야 함
# LSTM 쓰려면 무조건 3차원 자료여야 함

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1))) 
# 행무시 (사실상 면이지만 첫번째 숫자를 가리킴), 장황하게 설명했지만 코딩하면 한 줄
# LSTM은 3차원을 받아들임
model.add(Dense(20)) # 레이어의 첫번째 숫자는 아웃풋이야
model.add(Dense(10)) # Dense 레이어는 인풋은 1차원, 데이터는 2차원 받음
model.add(Dense(1)) # y는 하나

model.summary()

#3. compile and fit
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)

x_pred = np.array([5, 6, 7]) # (3,)
x_pred = x_pred.reshape(1, 3, 1) # 데이터 요소만 바뀌지 않고 LSTM에 쓸 수 있는 구조로 바뀔 뿐

#4. evaluate and predict
loss = model.evaluate(x, y)
print(loss)

result = model.predict(x_pred)
print(result)

# 결과 LSTM1
# 0.01851184293627739
# [[8.077794]]

# 예시와 함께 이해하는 LSTM의 구조
# 시계열 모델이 가장 잘 쓰이는 것이 주가예측이다
# 삼성전자의 주가가 다음과 같다고 해보자
# 1/1 : 80,000
# 1/2 : 79,000
# 1/3 : 82,000
# 1/4 : 85,000
# 1/5 : 78,000
# 1/6 : 75,000
# 1/7 : 79,000
# 이 때, 1/8의 주가를 예측하고 싶다
# 그렇다면 우리는 3일치를 끊고 싶고, 하나씩 연산을 하려고 하므로 X를 다음과 같이 구성해야 한다:
# X                      | Y
# [[[80], [79], [82]],   | 85
#  [[79], [82], [85]],   | 78
#  [[82], [85], [78]],   | 75
#  [[85], [78], [75]],   | 69
#  [[78], [75], [79]]]   | ?
# 따라서 훈련에 쓰일 X.shape은 (4, 3, 1)이 되고, 마지막 줄은 예측에 쓰인다.
# 이처럼 시계열 모델에서 Y의 값은 우리가 만들어줘야 한다.
# 마찬가지 방식으로 4일치로 끊어서 데이터를 구성할 수도 있다.
# 하지만 '도박사의 오류'를 조심해야 한다

# LSTM에 shape가 몇 차원? 3차원.
# LSTM 구조
# 따라해: 행, 열, 몇개씩 자르는지 (작업할 크기)


# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 10)                480
# _________________________________________________________________
# dense (Dense)                (None, 20)                220
# _________________________________________________________________
# dense_1 (Dense)              (None, 10)                210
# _________________________________________________________________
# dense_2 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 921
# Trainable params: 921
# Non-trainable params: 0
# _________________________________________________________________

# RNN parameters number 공식
# g * (i + h + 1) * h
# g, no. of FFNNs in a unit (RNN has 1, GRU has 3, LSTM has 4) # 3개의 sigmoid 함수, 1개의 tanh 함수 => 4개의 gate
# h, size of hidden units
# i, dimension/size of input

'''
예시로 설명하는 공식
어떤 한 LSTM 레이어를 다음과 같은 설정으로 만들었다고 가정하자:
input_dim=3
gate=4 (LSTM)
hidden_node (or output)=5

이 레이어의 파라미터 개수는 다음과 같이 두가지 계산을 통해 구할 수 있다.
1. 첫번째 계산
(i1)    --->   L    ---> (h1)
(i2)    --->   S    ---> (h2)
(i3)    --->   T    ---> (h3)
(b)     --->   M    ---> (h4)
                    ---> (h5)
위 순 방향 경우의 수는: (3 + 1) * 4 * 5 이다.

2. 두번째 계산
그리고 LSTM이 output을 소 되새김질 하듯이 다시 사용하는 모델이므로,
(h1)    --->   L    ---> (h1)
(h2)    --->   S    ---> (h2)
(h3)    --->   T    ---> (h3)
(h4)    --->   M    ---> (h4)
(h5)    --->        ---> (h5)
이러한 경우의 수를 곱하게 된다, 5 * 4 * 5.

따라서 이 모든 경우의 수를 정리하면,
(i + 1) * g * h     ... (가)
h * g * h           ... (나)
g * (i + h + 1) * h ... (가 + 나)
우리가 원하는 공식이 도출된다
'''

# LSTM Call arguments 중 
# inputs: A 3D tensor with shape (batch_size, timesteps, features 혹은 input_dim). 소위 "행, 열, 몇 개씩 자르는지"
# RNN (LSTM, GRU, SimpleRNN)의 activation='tanh' 디폴트 값 (hyperbolic tangent)

# SimpleRNN의 한계:
# 훈련이 데이터 후반에 영향을 줘서 성능을 높여주지만, 앞의 연산은 훈련의 효과를 못 봄
# LSTM은 이런 한계점을 보완한 것
# GRU는 LSTM에서 cell state을 빼서 더 가볍게 해준 것
# 내 모델이 삐꾸가 되지 않았을까? 했을 때 확인하는 것
# 나보다 구글 엔지니어가 잘 만들겠지? 남의 코드 확인하기 위해 보는 것
import numpy as np
import tensorflow as tf

#1. 데이터 준비
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1,activation='linear'))
model.add(Dense(3, activation='linear'))
model.add(Dense(4))
model.add(Dense(1))

model.summary()

'''
Model: "sequential" # 함수형일 때 "functional"
_________________________________________________________________
Layer (type)                 Output Shape              Param #    #(이전 노드 weight 개수+1(bias))*(이후 노드 weight 개수); parameter는 노드와 노드 사이의 연산 갯수
=================================================================
dense (Dense)                (None, 5) #행무시,노드갯수 10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 5
=================================================================
Total params: 49
Trainable params: 49
Non-trainable params: 0
_________________________________________________________________
'''

# 실습 + 과제
# ensemble 1, 2, 3, 그리고 bifurcation에 대해 서머리를 계산하고
# 이해한 것을 과제로 제출할 것

# layer 만들 때 'name' 파라미터에 대해 확인하고 설명할 것
# layer 만들 때 'name' 파라미터가 반드시 써야할 때가 있는데 설명해라
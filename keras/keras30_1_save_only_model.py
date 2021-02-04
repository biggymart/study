# 모델을 저장해보자, 저장하는 이유는?
# 나중에 재사용하기 위해서! (가중치 저장은 아직)

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

#2. model
model = Sequential()
model.add(LSTM(200, input_shape=(4,1)))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.summary()
'''
# 모델만 만들어도 summary 됨
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 200)               161600
_________________________________________________________________
dense (Dense)                (None, 100)               20100
_________________________________________________________________
dense_1 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_2 (Dense)              (None, 20)                1020
_________________________________________________________________
dense_3 (Dense)              (None, 10)                210
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 187,991
Trainable params: 187,991
Non-trainable params: 0
_________________________________________________________________'''

# model.save("path")
model.save("../data/h5/save_keras35.h5")
model.save("..//data//h5//save_keras35_1.h5")
model.save("..\data\h5\save_keras35_2.h5")
model.save("..\\data\\h5\\save_keras35_3.h5")
# (1) . 현재 폴더, 비주얼 스튜디오에서는 study가 현재 폴더, pycharm은 해당 file이 있는 폴더가 현재 폴더; (2) 당분간 저장용 확장자는 h5
# 실행해보면 모델 폴더 아래 "save_keras35.h5"라는 파일이 생성됨

# 경로 형식 (/, //, \, \\) 네 가지 다 사용 가능, 다만 escape code (e.g. \n) 같은 것을 주의해야 한다
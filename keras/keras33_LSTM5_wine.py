# scikitlearn, LSTM으로 모델링하고 Dense와 성능 비교하라 (다중분류, shuffle 반드시 필요)
# RNN계열 사용해도 무관 (SimpleRNN, GRU)

# keras22_3_wine.py 카피
#1. data
import numpy as np
from sklearn.datasets import load_wine
dataset = load_wine() # sklearn은 클래스의 인스턴스 만드는 식
x = dataset.data
y = dataset.target

#1-1. preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from tensorflow.keras.utils import to_categorical # OneHotEncoding from tensorflow
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1) # LSTM용 데이터로 가공
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(13, 1)))
model.add(Dense(3, activation='softmax')) # print(y) 에서 3가지로 분류된 것 확인함

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=3, mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류에서 loss는 반드시 categorical_crossentropy
model.fit(x_train, y_train, epochs=10000, callbacks=[early_stopping])

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("[categorical_crossentropy, acc] :", loss)

y_pred = model.predict(x_test[-5:])
for i in y_pred:
    print("(인덱스) 와인이름 :", np.argmax(i), dataset.target_names[np.argmax(i)], ", 값 :", np.max(i))
print(y_test[-5:])

# 결과 keras22_3_wine.py
# Epoch 1730/10000
# [categorical_crossentropy, acc] : [0.11018853634595871, 0.9722222089767456]
# (인덱스) 와인이름 : 0 class_0 , 값 : 1.0
# (인덱스) 와인이름 : 2 class_2 , 값 : 0.99999976
# (인덱스) 와인이름 : 0 class_0 , 값 : 0.9999982
# (인덱스) 와인이름 : 1 class_1 , 값 : 0.9998067
# (인덱스) 와인이름 : 0 class_0 , 값 : 0.99287087
# [[1. 0. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [1. 0. 0.]]

# 결과 keras33_LSTM5_wine.py
# Epoch 134/10000
# [categorical_crossentropy, acc] : [0.40608155727386475, 0.8888888955116272]
# (인덱스) 와인이름 : 0 class_0 , 값 : 0.9773424
# (인덱스) 와인이름 : 2 class_2 , 값 : 0.52174264
# (인덱스) 와인이름 : 0 class_0 , 값 : 0.88920027
# (인덱스) 와인이름 : 1 class_1 , 값 : 0.9241515
# (인덱스) 와인이름 : 2 class_2 , 값 : 0.46973544
# [[1. 0. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [1. 0. 0.]]
# keras22_cancer1.py 를 다중분류로 코딩하시오
# 이중분류는 다중분류지만, 다중분류는 이중분류가 아니다

# OneHotEncoding: 이진분류와 구분되는 특징 (x와 y의 차원 모두 2차원, y를 벡터화 != 'y에 대한 전치리')
# Output Layer node == number of classification

import numpy as np
from sklearn.datasets import load_breast_cancer

#1. data
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# preprocessing (train_test_split, OneHotEncoding, MinMaxScaler)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
# ** 1st difference btwn a binary and a categorical **

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
for i in range(10):
    model.add(Dense(10, activation='relu', input_shape=(30,))) 
model.add(Dense(2, activation='softmax')) 
# ** 2nd difference (num of nodes == num of features) **

#3. compile and fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# ** 3rd difference (loss value) **

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=3, mode='min') 

model.fit(x_train, y_train, epochs=10000, validation_split=0.2, callbacks=[early_stopping])

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("loss, metrics :", loss)

y_predict = model.predict(x_test[-20:])
for i in y_predict:
    print(np.argmax(i), end=' ')
print()
print(y[-20:])

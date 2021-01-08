# scikitlearn, LSTM으로 모델링하고 Dense와 성능 비교하라 (이진분류)

# keras21_cancer1.py 카피
#1. data
import numpy as np
from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

#1-1. preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

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
model.add(LSTM(10, activation='relu', input_shape=(30,1)))
for i in range(10):
    model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10000, validation_split=0.2, callbacks=[early_stopping])

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("binary_crossentropy, acc :", loss)

y_predict = model.predict(x_test[-10:])
y_pred = list(map(int,np.round(y_predict,0)))
y_predict = np.transpose(y_predict)
y_pred = np.transpose(y_pred)
print(y_pred)
print(y_test[-10:])

# 결과 keras21_cancer1.py
# binary_crossentropy, acc : [0.11954214423894882, 0.9561403393745422]
# [0 1 1 1 0 1 0 1 1 1]
# [0 1 1 1 0 1 0 1 1 1]

# 결과 keras33_LSTM3_cancer.py
# binary_crossentropy, acc : [0.255013644695282, 0.9649122953414917]
# [0 1 1 1 0 1 0 1 1 1]
# [0 1 1 1 0 1 0 1 1 1]
# hist를 이용하여 그래프를 그리시오 (loss, val_loss, acc, val_acc)
# 이진분류 cancer --> keras36_hist1_cancer.py 에 그려보자

# keras21_cancer1.py 카피
import numpy as np
from sklearn.datasets import load_breast_cancer

#1. data
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

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(30,)))
for i in range(10):
    model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min') 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, batch_size=8, epochs=10000, validation_split=0.2, callbacks=[early_stopping])
# graph 그리기 위해서 변수로 반환

#4. evaluate and predict
loss = model.evaluate(x_test, y_test, batch_size=8)
print("binary_crossentropy, acc :", loss)

y_predict = model.predict(x_test[-10:])
y_pred = list(map(int,np.round(y_predict,0)))
y_predict = np.transpose(y_predict)
y_pred = np.transpose(y_pred)
print(y_pred)
print(y_test[-10:])

# graph
print(hist.history.keys()) # dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss']) # train loss
plt.plot(hist.history['val_loss']) # val loss
plt.plot(hist.history['acc']) # train acc
plt.plot(hist.history['val_acc']) # val acc

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()

# 결과
# binary_crossentropy, acc : [0.06318700313568115, 0.9824561476707458]
# [0 1 1 1 0 1 0 1 1 1]
# [0 1 1 1 0 1 0 1 1 1]
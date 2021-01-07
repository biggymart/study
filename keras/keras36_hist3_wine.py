# 와인의 품종을 확인하는 데이터셋
# 실습> DNN 모델을 완성시켜라

# keras22_3_wine.py 카피
#1. data
import numpy as np
from sklearn.datasets import load_wine
dataset = load_wine()
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

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(13,)))
model.add(Dense(3, activation='softmax'))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=3, mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, batch_size=4, epochs=10000, validation_split=0.2, callbacks=[early_stopping])
# graph 그리기 위해서 변수로 반환

#4. evaluate and predict
loss = model.evaluate(x_test, y_test, batch_size=4)
print("[categorical_crossentropy, acc] :", loss)

y_pred = model.predict(x_test[-5:])
for i in y_pred:
    print("(인덱스) 와인이름 :", np.argmax(i), dataset.target_names[np.argmax(i)], ", 값 :", np.max(i))
print(y_test[-5:])

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
# [categorical_crossentropy, acc] : [0.05800100043416023, 0.9722222089767456]
# (인덱스) 와인이름 : 0 class_0 , 값 : 0.99997413
# (인덱스) 와인이름 : 2 class_2 , 값 : 0.99974364
# (인덱스) 와인이름 : 0 class_0 , 값 : 0.9989079
# (인덱스) 와인이름 : 1 class_1 , 값 : 0.9724035
# (인덱스) 와인이름 : 0 class_0 , 값 : 0.8666358
# [[1. 0. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [1. 0. 0.]]
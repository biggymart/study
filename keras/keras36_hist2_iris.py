# hist를 이용하여 그래프를 그리시오 (loss, val_loss, acc, val_acc)
# 다중분류 iris, wine --> keras36_hist2_iris.py, keras36_hist3_wine.py 에 그려보자

# keras22_1_iris1_keras.py 카피
import numpy as np
from sklearn.datasets import load_iris

#1. data
dataset = load_iris()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# OneHotEncoding, tensorflow
from tensorflow.keras.utils import to_categorical 
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
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(3, activation='softmax'))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=3, mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=10000, validation_split=0.2, callbacks=[early_stopping])
# graph 그리기 위해서 변수로 반환

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print(loss)

y_pred = model.predict(x_test[-5:])
for i in y_pred:
    print("(인덱스) 꽃이름 :", np.argmax(i), dataset.target_names[np.argmax(i)], ", 값 :", np.max(i))
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
# [0.11358293145895004, 1.0]
# (인덱스) 꽃이름 : 2 virginica , 값 : 0.998154
# (인덱스) 꽃이름 : 0 setosa , 값 : 0.9992379
# (인덱스) 꽃이름 : 0 setosa , 값 : 0.94172317
# (인덱스) 꽃이름 : 2 virginica , 값 : 0.5677543
# (인덱스) 꽃이름 : 2 virginica , 값 : 0.9256756
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]]
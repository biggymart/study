# scikitlearn, LSTM으로 모델링하고 Dense와 성능 비교하라 (다중분류, shuffle 반드시 필요)

# keras22_1_iris1_keras.py 카피
import numpy as np
from sklearn.datasets import load_iris

#1. data
dataset = load_iris()
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
model.add(LSTM(10, activation='relu', input_shape=(4,1)))
model.add(Dense(3, activation='softmax'))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=3, mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10000, callbacks=[early_stopping])

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print(loss)

y_pred = model.predict(x_test[-5:])
for i in y_pred:
    print("(인덱스) 꽃이름 :", np.argmax(i), dataset.target_names[np.argmax(i)], ", 값 :", np.max(i))
print(y_test[-5:])

# 결과 keras22_1_iris1_keras.py
# [0.12124720960855484, 0.9666666388511658]
# (인덱스) 꽃이름 : 2 virginica , 값 : 0.9984915
# (인덱스) 꽃이름 : 0 setosa , 값 : 0.9997532
# (인덱스) 꽃이름 : 0 setosa , 값 : 0.94546694
# (인덱스) 꽃이름 : 1 versicolor , 값 : 0.5085082
# (인덱스) 꽃이름 : 2 virginica , 값 : 0.94557166
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]]

# 결과 keras33_LSTM4_iris.py
# [0.15329107642173767, 1.0]
# (인덱스) 꽃이름 : 2 virginica , 값 : 0.99720263
# (인덱스) 꽃이름 : 0 setosa , 값 : 0.9827596
# (인덱스) 꽃이름 : 0 setosa , 값 : 0.9793773
# (인덱스) 꽃이름 : 2 virginica , 값 : 0.55923414
# (인덱스) 꽃이름 : 2 virginica , 값 : 0.8397623
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]]
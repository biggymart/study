# 과제2> 원핫인코딩 tensorflow / keras 에서 제공되는 to_categorical 를 sklearn에서 제공되는 것으로 바꿀 것 

import numpy as np
from sklearn.datasets import load_iris

#1. data
dataset = load_iris()
x = dataset.data
y = dataset.target

# OneHotEncoding, sklearn
y = y.reshape(-1, 1)

from sklearn.preprocessing import OneHotEncoder #LabelEncoder (만약 y 데이터가 0, 1 ,2 등 이런 식으로 분류되어 있지 않으면 써야함)
'''
y = y.reshape(-1,1)
enc = OneHotEncoder()
enc.fit(y)
y = enc.transform(y).toarray()
'''
enc = OneHotEncoder()
y = enc.fit_transform(y.reshape(-1,1)).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(3, activation='softmax'))

#3. compile and fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=3, mode='min')

model.fit(x_train, y_train, validation_split=0.2, epochs=10000, callbacks=[early_stopping])

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print(loss)

y_pred = model.predict(x_test[-5:])
for i in y_pred:
    print(np.max(i), np.argmax(i))
print(y_test[-5:])

# 결과 keras22_1_iris1_keras
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

# 결과 keras22_1_iris2_sklearn
# [0.14184093475341797, 0.9666666388511658]
# (인덱스) 꽃이름 : 2 virginica , 값 : 0.9974005
# (인덱스) 꽃이름 : 0 setosa , 값 : 0.99797815
# (인덱스) 꽃이름 : 0 setosa , 값 : 0.95333207
# (인덱스) 꽃이름 : 1 versicolor , 값 : 0.6185203
# (인덱스) 꽃이름 : 2 virginica , 값 : 0.9339602
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]]
# keras38_dropout4_iris.py 카피

#1. data
import numpy as np
from sklearn.datasets import load_iris
dataset = load_iris()
x = dataset.data
y = dataset.target

from sklearn.preprocessing import OneHotEncoder # OneHotEncoding, sklearn
enc = OneHotEncoder()
y = enc.fit_transform(y.reshape(-1,1)).toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=3, mode='min')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, validation_split=0.2, epochs=10000, callbacks=[early_stopping])

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print(loss)

y_pred = model.predict(x_test[-5:])
for i in y_pred:
    print(np.max(i), np.argmax(i))
print(y_test[-5:])

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

# 결과 Dropout 후
# [1.1363680362701416, 0.06666667014360428]
# 0.41082457 2
# 0.4094279 2
# 0.34564963 2
# 0.39012042 0
# 0.3983568 0
# [[0. 0. 1.]
#  [1. 0. 0.]
#  [1. 0. 0.]
#  [0. 0. 1.]
#  [0. 0. 1.]]

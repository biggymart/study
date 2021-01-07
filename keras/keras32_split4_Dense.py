# 과제 및 실습 (Dense 모델) + keras32_split3_LSTM.py와 결과비교
# 전처리, EarlyStopping 등등 다 포함
# 데이터 1~100 /
# x                 y
# 1,2,3,4,5         6
# ...
# 95,96,97,98,99    100

# predict를 만들 것
# 96,97,98,99,100   101
# ...
# 100,101,102,103,104   105
# 예상 predict는 (101,102,103,104,105) 

# 제공된 데이터
import numpy as np
a = np.array(range(1,101))
b = np.array(range(96,106))
size = 6


# keras31_split.py 함수
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)] 
        aaa.append(subset) # aaa.append([item for item in subset])와 같음
    return np.array(aaa)

#1. data
dataset = split_x(a, size)
predictset = split_x(b, size)
x = dataset[:, :5] # (100,5) array[row, column]
y = dataset[:, 5] # (100,)

x_pred = predictset[:, :5]
y_pred = predictset[:, 5]

#1-1. preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

#2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(5,)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=1000, batch_size=4, callbacks=[early_stopping], validation_split=0.2)

#4. evaluate and predict
loss = model.evaluate(x_test, y_test, batch_size=4)
print("loss(mse, mae) :", loss)

y_pred = model.predict(x_pred)
print("y_predict :", y_pred)

# 결과 keras32_split3_LSTM
# loss(mse, mae) : [0.22615131735801697, 0.3781367838382721]
# y_predict : 
# [[102.59343 ]
#  [103.81069 ]
#  [105.0351  ]
#  [106.26664 ]
#  [107.505165]]

# 결과 keras32_split4_Dense
# loss(mse, mae) : [1.933873026993549e-11, 3.1119898267206736e-06]
# y_predict : 
# [[101.00001]
#  [102.     ]
#  [103.00001]
#  [104.     ]
#  [105.     ]]
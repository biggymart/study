# Dense 모델을 구성하시오

# 제공된 데이터
import numpy as np
a = np.array(range(1,11))
size = 5

# keras31_split.py 함수
def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)] 
        aaa.append(subset) # aaa.append([item for item in subset])와 같음
    return np.array(aaa)

#1. data
dataset = split_x(a, size)
x = dataset[:, :4] # (6,4) array[row, column]
y = dataset[:, 4] # (6,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_pred = np.array([7, 8, 9, 10]) # (4,)
x_pred = x_pred.reshape(1, -1)
x_pred = scaler.transform(x_pred)

#2. model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, callbacks=[early_stopping])

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("loss(mse) :", loss)

y_predict = model.predict(x_pred)
print("y_predict :", y_predict) # 예상값 11

# 결과 keras32_split1_LSTM
# loss(mse) : 0.009372422471642494
# y_predict : [[11.276264]]

# 결과 keras32_split2_Dense
# loss(mse) : 6.934897101018578e-12
# y_predict : [[11.000005]]
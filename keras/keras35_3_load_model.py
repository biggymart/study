# 데이터를 넣어서 load_model 한 번 써보자
# (1)과 (2)를 활용하여 코드를 완성하여라

#1. data
# (1) 주어진 데이터
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

dataset = split_x(a, size)
x = dataset[:, :4] # (6,4) array[row, column]
y = dataset[:, 4]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1) # LSTM용 데이터로 가공
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
 
x_pred = np.array([8, 9, 10, 11]) # (4,)
x_pred = x_pred.reshape(1, -1)
x_pred = scaler.transform(x_pred)
x_pred = x_pred.reshape(1, 4, 1)

#2. model
###  (2) 활용할 부분 ###
from tensorflow.keras.models import load_model
model = load_model('./model/save_keras35.h5')

# model.summary()
# 제대로 된다면 1 파일에서 만든 모델이 구동이 된다는 것이고, 모델에 맞게 데이터의 shape을 맞춰야한다
# 즉, summary를 보고 데이터를 잘 짤라야 한다

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, callbacks=[early_stopping])

#4. evaluate and predict
loss = model.evaluate(x_test, y_test)
print("loss(mse) :", loss)

y_predict = model.predict(x_pred)
print("y_predict :", y_predict)

# 결과
# loss(mse) : 3.8466085243271664e-05
# y_predict : [[11.930509]]
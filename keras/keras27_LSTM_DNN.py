# keras23_LSTM3_scale 을 DNN으로 코딩
# 결과치 비교 / 23파일보다 loss 값이 더 낮게 만들 것

#1. data
import numpy as np
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

# print(x.shape) #(13, 3)
# print(y.shape) #(1,)

#1-1. preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=3)) # input_shape 은 1차원
model.add(Dense(32))
model.add(Dense(1))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[early_stopping])


#4. evaluate and predict
loss = model.evaluate(x_test, y_test, batch_size=1)
print("loss(mse) :", loss)

x_pred = np.array([50,60,70]) # (3,)
x_pred = x_pred.reshape(1, -1)
x_pred = scaler.transform(x_pred)
result = model.predict(x_pred)
print(result)

# 결과 LSTM3
# loss(mse) : 1.0227223634719849
# [[80.15487]]

# 결과 DNN
# loss(mse) : 0.2535952925682068
# [[87.42364]]
# keras23_LSTM3_scale.py 카피
# conv1D를 활용하여 코드를 완성하시오 (과제: summary 정리)

#1. data
import numpy as np
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
# x.shape: (13, 3)
# y.shape: (13,)
'''
np.save('../data/npy/k54_conv1d_01_lstm_x.npy', arr=x)
np.save('../data/npy/k54_conv1d_01_lstm_y.npy', arr=y)

x_data = np.load('../data/npy/k54_conv1d_01_lstm_x.npy')
y_data = np.load('../data/npy/k54_conv1d_01_lstm_y.npy')
'''
#1-1. preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1) # conv1D => (N, row, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout

model = Sequential()
model.add(Conv1D(6, 3, activation='relu', padding='valid', input_shape=(3,1)))
# model.add(MaxPool1D(2))
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelCheckpoint/k54_conv1d_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
es = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(loss='mse', optimizer='adam', metrics='mae')
model.fit(x_train, y_train, epochs=1000, batch_size=1, callbacks=[es, cp], validation_split=0.2)

# model.save('../data/h5/k54_conv1d_01_lstm.h5')

#4. evaluate and predict
loss = model.evaluate(x_test, y_test, batch_size=1)
print("loss(mse, mae) :", loss)

x_pred = np.array([50,60,70])# (3,)
x_pred = x_pred.reshape(-1, 3)
x_pred = scaler.transform(x_pred)
x_pred = x_pred.reshape(-1, x_pred.shape[1], 1)

result = model.predict(x_pred)
print(result)

# 결과 keras23_LSTM3_scale.py
# loss(mse) : 0.10934270173311234
# [[78.521965]]

# 결과 keras54_conv1d_01_lstm.py
# loss(mse, mae) : [56.709407806396484, 7.410697937011719]
# [[[80.02453]]]

# model.summary()
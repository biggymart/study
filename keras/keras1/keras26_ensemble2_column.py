# keras30_ensemble6_row 카피
# 앙상블은 열이 달라도 가능

#1. data
import numpy as np

x1 = np.array([[1,2],[2,3],[3,4],[4,5],
            [5,6],[6,7],[7,8],[8,9],
            [9,10],[10,11],
            [20,30],[30,40],[40,50]]) # (13, 2)
x2 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]]) # (13, 3)

y1 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]]) # (13, 3)
y2 = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70]) # (13,)

#1-1. preprocessing (train_test_split, MinMaxScaler, reshape)
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8)

from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
scaler1.fit(x1_train)
x1_train = scaler1.transform(x1_train)
x1_test = scaler1.transform(x1_test)

scaler2 = MinMaxScaler()
scaler2.fit(x2_train)
x2_train = scaler2.transform(x2_train)
x2_test = scaler2.transform(x2_test)

# LSTM용 데이터로 가공
x1_train = x1_train.reshape(x1_train.shape[0], x1_train.shape[1], 1)
x1_test = x1_test.reshape(x1_test.shape[0], x1_test.shape[1], 1)
x2_train = x2_train.reshape(x2_train.shape[0], x2_train.shape[1], 1)
x2_test = x2_test.reshape(x2_test.shape[0], x2_test.shape[1], 1)

# print(x.shape) # (13, 3, 1)
# print(x.shape) # (13, 3, 1)
# print(y.shape) # (13,)

x1_predict = np.array([55,65]) # (2,)
x1_predict = x1_predict.reshape(1, -1)
x1_predict = scaler1.transform(x1_predict)
x1_pred = x1_predict.reshape(1, 2, 1)

x2_predict = np.array([65,75,85]) # (3,)
x2_predict = x2_predict.reshape(1, -1)
x2_predict = scaler2.transform(x2_predict)
x2_pred = x2_predict.reshape(1, 3, 1)

#2. model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate

input1 = Input(shape=(2, 1))
lstm1 = LSTM(32, activation='relu')(input1)
dense1 = Dense(16, activation='relu')(lstm1)
dense1 = Dense(8, activation='relu')(dense1)
dense1 = Dense(8, activation='relu')(dense1)
dense1 = Dense(8, activation='relu')(dense1)

input2 = Input(shape=(3, 1))
lstm2 = LSTM(32, activation='relu')(input2)
dense2 = Dense(16, activation='relu')(lstm2)
dense2 = Dense(10, activation='relu')(dense2)
dense2 = Dense(10, activation='relu')(dense2)
dense2 = Dense(10, activation='relu')(dense2)

merge1 = concatenate([dense1, dense2])
# middle1 = Dense(32)(merge1) # optional layer
output1 = Dense(32)(merge1)
output1 = Dense(16)(output1)
output1 = Dense(16)(output1)
output1 = Dense(16)(output1)
output1 = Dense(1)(output1)

output2 = Dense(32)(merge1)
output2 = Dense(16)(output2)
output2 = Dense(16)(output2)
output2 = Dense(16)(output2)
output2 = Dense(1)(output2)

model = Model(inputs=[input1, input2], outputs=[output1,output2])
# model.summary()

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='min')

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train],epochs=1000, batch_size=1, validation_split=0.2, verbose=1, callbacks=[early_stopping])

#4. evaluate and predict
result = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print(model.metrics_names, ":", result)

y1_predict = model.predict([x1_pred, x2_pred])
print("y1_predict :", y1_predict)

# 결과 keras29_LSTM_ensemble
# 'loss', 'mae':[0.04449871554970741, 0.15008099377155304]
# y1_predict : [[80.57303]]


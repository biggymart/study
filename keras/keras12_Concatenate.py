# concatenate를 Concatenate로 바꿔라

#1. data
import numpy as np
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
y1 = np.array([range(711,811), range(1,101), range(201, 301)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1); x2 = np.transpose(x2)
y1 = np.transpose(y1); y2 = np.transpose(y2)

from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=0.8, shuffle=False)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size=0.8, shuffle=False)

#2. model
# 인풋 모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Concatenate

input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)

input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)

# 모델 병합
merge1 = Concatenate()([dense1, dense2]) # concatenate([dense1, dense2]), 문법만 다를 뿐 기능은 똑같음
middle1 = Dense(30)(merge1)
middle1 = Dense(10)(middle1)
middle1 = Dense(10)(middle1)

# 아웃풋 모델
output1 = Dense(30)(middle1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(middle1)
output2 = Dense(7)(output2)
output2 = Dense(7)(output2)
output2 = Dense(3)(output2)

# 모델 선언
model = Model(inputs=[input1, input2], outputs=[output1, output2])

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=1, validation_split=0.2, verbose=1)

#4. evaluate and predict
loss = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)
print( 'loss :', loss)
print("model.metrics_names :", model.metrics_names)

y1_predict, y2_predict = model.predict([x1_test, x2_test])

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', (RMSE(y1_test, y1_predict) + RMSE(y2_test, y2_predict))/int(len(loss)/2))

# R2 구하기
from sklearn.metrics import r2_score
print('R2 :', (r2_score(y1_test, y1_predict) + r2_score(y2_test, y2_predict))/int(len(loss)/2))
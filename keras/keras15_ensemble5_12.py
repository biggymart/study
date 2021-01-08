# 1:다 분기모델
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input


#1. data
x1 = np.array([range(100), range(301, 401), range(1, 101)])
y1 = np.array([range(711,811), range(1,101), range(201, 301)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1); y2 = np.transpose(y2)

x1_train, x1_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, y1, y2, train_size=0.8, shuffle=False)

#2. model
# input model
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)
dense1 = Dense(10, activation='relu')(dense1)

# output model
output1 = Dense(30)(dense1)
output1 = Dense(30)(output1)
output1 = Dense(30)(output1)
output1 = Dense(3)(output1)

output2 = Dense(15)(dense1)
output2 = Dense(30)(output2)
output2 = Dense(30)(output2)
output2 = Dense(3)(output2)

# 모델 선언
model = Model(inputs=input1, outputs=[output1, output2])

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x1_train, [y1_train, y2_train], epochs=100, batch_size=1, validation_split=0.2, verbose=0)

#4. evaluate and predict
loss = model.evaluate(x1_test, [y1_test, y2_test], batch_size=1)
print("model.metrics_names :", model.metrics_names)
print('loss :', loss)

y1_predict, y2_predict = model.predict(x1_test)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', (RMSE(y1_test, y1_predict) + RMSE(y2_test, y2_predict))/int(len(loss)/2))

# R2 구하기
from sklearn.metrics import r2_score
print('R2 :', (r2_score(y1_test, y1_predict) + r2_score(y2_test, y2_predict))/int(len(loss)/2))

# 결과
# model.metrics_names : ['loss', 'dense_7_loss', 'dense_11_loss', 'dense_7_mae', 'dense_11_mae']
# loss : [0.029719442129135132, 0.01543012447655201, 0.014289319515228271, 0.09844067692756653, 0.09299774467945099]
# RMSE : 0.12189389623027255
# R2 : 0.9995529765134821
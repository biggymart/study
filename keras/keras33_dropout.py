# keras18_EarlyStopping.py 카피

#1. data
import numpy as np
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
_, x_val, _, y_val = train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

# 실습: 모델을 구성하시오
#2. model
'''
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
input1 = Input(shape=(13,))
dense1 = Dense(128, activation='relu')(input1)
dropout1 = Dropout(0.2)(dense1)
dense1 = Dense(64, activation='relu')(dropout1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1, outputs=output1)
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(128, input_dim=13))
model.add(Dropout(0.1)) # 0.1 ~ 0.5까지 씀
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(1))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
hist = model.fit(x_train, y_train, batch_size=4, epochs=2000, verbose=1, validation_data=(x_val, y_val), callbacks=[early_stopping])

#4. evalutate and predict
mse, mae = model.evaluate(x_test, y_test, batch_size=4)
print("mse :", mse, "\nmae :", mae)

y_predict = model.predict(x_test)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2 :', r2)

# graph
print(hist.history.keys()) # dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])

import matplotlib.pyplot as plt
plt.plot(hist.history['loss']) # train loss
plt.plot(hist.history['val_loss']) # val loss
plt.plot(hist.history['mae']) # train mae
plt.plot(hist.history['val_mae']) # val mae

plt.title('loss & mae')
plt.ylabel('loss, mae')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss', 'train mae', 'val mae'])
plt.show()

# 결과 keras18_EarlyStopping, Dropout 전
# Epoch 214/2000
# mse : 8.512691497802734
# mae : 1.930296778678894
# RMSE : 2.917651639027537
# R2 : 0.90123263762414

# 결과 Dropout 후
# Epoch 54/2000
# mse : 32.51150131225586 
# mae : 3.8910696506500244
# RMSE : 5.701885751596799
# R2 : 0.6522116342653763

# Dropout 후 성능이 저하된 것으로 보아 불필요한 노드는 없음을 알 수 있음
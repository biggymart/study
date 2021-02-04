# boston, diabetes, cancer, iris, wine 을 Conv2D으로 구현해라
# Tip> data shape을 4차원으로 맞추시오.

# keras38_dropout1_boston.py 카피

#1. data
import numpy as np
from sklearn.datasets import load_boston
dataset = load_boston()
x = dataset.data # (506, 13)
y = dataset.target # (506)

#1-0. preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(-1, x_train.shape[1], 1, 1)
x_test = x_test.reshape(-1, x_test.shape[1], 1, 1)

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
model = Sequential()
model.add(Conv2D(30, (2,2), 2, 'same', input_shape=(x.shape[1], 1, 1)))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(64))
model.add(Dropout(0.2)) # 0.1 ~ 0.5까지 씀
model.add(Dense(8))
model.add(Dense(1))
# model.summary()

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
hist = model.fit(x_train, y_train, batch_size=4, epochs=2000, validation_split=0.2, callbacks=[early_stopping])

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
# print(hist.history.keys()) # dict_keys(['loss', 'mae', 'val_loss', 'val_mae'])

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

# 결과 keras41_cnn1_boston.py
# Epoch 34/2000
# mse : 24.26955795288086
# mae : 3.7126715183258057
# RMSE : 4.926414549159304
# R2 : 0.7145757018419747
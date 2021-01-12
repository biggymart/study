# keras38_dropout1_boston.py 카피

import numpy as np
from sklearn.datasets import load_boston

#1. data
dataset = load_boston()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
_, x_val, _, y_val = train_test_split(x, y, test_size=0.2) # 전처리

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val) # 전처리

# 실습: 모델을 구성하시오
#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(256, input_dim=13))
#model.add(Dropout(0.1)) # 0.1 ~ 0.5까지 씀
model.add(Dense(64))
model.add(Dense(1))

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint ### Point1 ###
modelpath = './modelCheckPoint/k46_boston_{epoch:02d}-{val_loss:.4f}.hdf5' ### Point2 ###
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto') ### Point3 ###
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
hist = model.fit(x_train, y_train, batch_size=4, epochs=200, validation_data=(x_val, y_val), callbacks=[early_stopping, check_point])

#4. evalutate and predict
mse, mae = model.evaluate(x_test, y_test, batch_size=4)
print("mse :", mse, "\nmae :", mae)

y_predict = model.predict(x_test) # 단순히 지표를 만들기 위해 만든 변수임

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2 :', r2)

# Visualization ### Point5 ###
import matplotlib.pyplot as plt # Matplotlib 한글 미지원
plt.figure(figsize=(10, 6)) # 도화지 면적을 잡아줌, 가로가 10, 세로가 6

plt.subplot(2, 1, 1) # 2행 1열 짜리 그래프를 만들겠다, 그 중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2) # 2행 1열 그래프 중 두번째
plt.plot(hist.history['mae'], marker='.', c='red', label='mae')
plt.plot(hist.history['val_mae'], marker='.', c='blue', label='val_mae')
plt.grid() # 격자

plt.title('MAE')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

# 결과
# mse : 15.413471221923828
# mae : 3.072878360748291
# RMSE : 3.925999113505272
# R2 : 0.8083866637407635
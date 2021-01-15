# 과제> np.load를 활용하여 모델을 완성하시오

#1. data
import numpy as np
x_data = np.load('../data/npy/diabetes_x.npy')
y_data = np.load('../data/npy/diabetes_y.npy')

# keras46_MC_5_diabetes.py 카피
#1-1. preprocessing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=1)
_, x_val, _, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
x_val = scaler.transform(x_val)

#2. model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
input1 = Input(shape=(10,))
dense1 = Dense(200, activation='relu')(input1)
# dropout1 = Dropout(0.1)
dense1 = Dense(200, activation='relu')(dense1)
dense1 = Dense(200, activation='relu')(dense1)
dense1 = Dense(200, activation='relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1, outputs=output1)

#3. compile and fit
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = '../data/modelCheckpoint/k46_diabetes_{epoch:02d}-{val_loss:.4f}.hdf5'
check_point = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
hist = model.fit(x_train, y_train, batch_size=1, epochs=1000, validation_data=(x_val, y_val), callbacks=[early_stopping, check_point])

#4. evaluate and predict
mse, mae = model.evaluate(x_test, y_test, batch_size=1)
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

# Visualization
import matplotlib.pyplot as plt 
plt.figure(figsize=(10, 6))

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
# mse : 3525.645263671875
# mae : 42.262699127197266
# RMSE : 59.37715020153991
# R2 : 0.33840015117629896
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


#1. data
x = np.array([range(100), range(301, 401), range(1, 101), range(201, 301), range(401, 501)])
x = np.transpose(x)
y = np.array([range(711,811), range(1,101)])
y = np.transpose(y)

x_pred2 = np.array([100, 402, 101, 301, 501])
x_pred2 = x_pred2.reshape(1, 5)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. model
model = Sequential()
model.add(Dense(10, input_dim=5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(2))

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, verbose=0) 
# validation 의 default 가 없다 (None) 
# verbose=0 
# (Epoch 생략), 출력하는 burden이 줄어들어서 조금 더 빨리 됨

# verbose=1 (default) 
# Epoch 100/100 64/64 [==============================] - 0s 1ms/step - loss: 2.0989e-09 - mae: 3.2406e-05 - val_loss: 9.0589e-10 - val_mae: 1.9103e-05

# verbose=2 
# Epoch 100/100 64/64 - 0s - loss: 4.1369e-09 - mae: 4.6221e-05 - val_loss: 6.3747e-09 - val_mae: 6.7979e-05

# verbose=3
# Epoch 100/100


#4. evaluate and predict
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss, '\nmae :', mae)

y_predict = model.predict(x_test)
# print(y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', RMSE(y_test, y_predict))

# R2 구하기
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('R2 :', r2)

y_pred2 = model.predict(x_pred2)
print(y_pred2)
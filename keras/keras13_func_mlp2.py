import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split


#1. data
x = np.array([range(100), range(301, 401), range(1, 101)])
x = np.transpose(x)
y = np.array([range(711,811), range(1,101), range(100)])
y = np.transpose(y)
# shape == (100, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2. model
input1 = Input(shape=(3,))
dense1 = Dense(10)(input1)
dense2 = Dense(5)(dense1)
dense3 = Dense(5)(dense2)
output1 = Dense(3)(dense3)
model = Model(inputs=input1, outputs=output1)

'''
# keras09_mlp3.py
model = Sequential()
model.add(Dense(10, input_dim=3)) # feature 3개
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(3)) # output 3개
'''

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100 ,batch_size=1, validation_split=0.2)

#4. evaluate and predict
loss, mae = model.evaluate(x_test, y_test)
print( 'loss :', loss, "\nmae :", mae)

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

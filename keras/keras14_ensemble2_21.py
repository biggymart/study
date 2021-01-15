# 다:1 앙상블, y2 데이터 다 날려라
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate # Concatenate, LSTM, Conv2D


# 1. data
x1 = np.array([range(100), range(301, 401), range(1, 101)])
x2 = np.array([range(101, 201), range(411, 511), range(100, 200)])
y = np.array([range(711,811), range(1,101), range(201, 301)])

x1 = np.transpose(x1); x2 = np.transpose(x2); y = np.transpose(y)

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.8, shuffle=False)

# print(x1_train.shape, x1_test.shape, x2_train.shape, x2_test.shape, y_train.shape, y_test.shape)
# (80, 3) (20, 3) (80, 3) (20, 3) (80, 3) (20, 3)

# 2. model
# 2-1. first model
input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(5, activation='relu')(dense1)

# 2-2. second model
input2 = Input(shape=(3,))
dense2 = Dense(10, activation='relu')(input2)
dense2 = Dense(5, activation='relu')(dense2)
dense2 = Dense(5, activation='relu')(dense2)

# 2-3. model concatenate
merge1 = concatenate([dense1, dense2])
output1 = Dense(30)(merge1)
output1 = Dense(7)(output1)
output1 = Dense(3)(output1)

# 2-4. model declaration
model = Model(inputs=[input1, input2], outputs=output1)

# 3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1_train, x2_train], y_train, epochs=10, batch_size=1, validation_split=0.2, verbose=0)

# 4. evaluate and predict
loss = model.evaluate([x1_test, x2_test], y_test, batch_size=1)
print(model.metrics_names, loss)

y_predict = model.predict([x1_test, x2_test])
# print("y_predict :\n", y_predict)

# RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE (y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE :', (RMSE(y_test, y_predict)))

# R2 구하기
from sklearn.metrics import r2_score
print('R2 :', (r2_score(y_test, y_predict)))
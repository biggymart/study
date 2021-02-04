from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. data
x = np.array(range(1, 101))
y = np.array(range(1, 101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, shuffle=True)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(1))

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2)
# 80개 중에 20퍼센트라서 validation data는 16개

#4. evaluate and predict
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss, 'mae :', mae)

y_predict = model.predict(x_test)
print('y_predict :', y_predict)

# validation = 0.2
# loss : 3.717044091899879e-05 mae : 0.005014392547309399

# 요약
# 1. fit에 validation_split=flaot_value 활용하여 data를 split 하는 방식
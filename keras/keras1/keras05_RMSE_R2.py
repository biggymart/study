from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # y = wx + b
import numpy as np


#1. data
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x_test = np.array([11, 12, 13, 14, 15])
y_test = np.array([11, 12, 13, 14, 15])
x_pred = np.array([16, 17, 18])

#2. model
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae']) 
# metrics=['accuracy'] 를 쓰려면 회귀가 아니라 분류 모델이어야 함
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2) 

#4. evaluate and predict
results = model.evaluate(x_test, y_test, batch_size=1)
print("mse, mae :", results)

y_predict = model.predict(x_test)
# print("y_predict :", y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict): # 여러번 재사용하기 위해서 함수를 사용한다
    return np.sqrt(mean_squared_error(y_test, y_predict))  # 매개변수 1: 실제값, 매개변수 2: 예측값
    # sklearn에서는 mse를 이렇게 구할 수 있다
    # np.sqrt(result[0]) 도 가능함
print("RMSE :", RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 :", r2) # accuracy 와는 명확히 다른 지표; 1.0이 제일 좋음
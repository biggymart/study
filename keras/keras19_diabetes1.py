# 실습: keras18을 참고하여 총 6가지의 버전을 만드시오
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

#1. data
dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8)

'''
# 데이터 파악하기
print(x[:5]) # 일부 데이터를 보고 이 모델이 회귀일지 분류일지 파악
print(y[:10])
print(x.shape, y.shape) # (442, 10) (442,)

print(np.max(x), np.min(x)) # 전처리가 되었는지 파악하기, 안 되었네
print(dataset.feature_names) # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'], 10개 features
print(dataset.DESCR) # 데이터 설명서 불러오기
'''

#2. model
input1 = Input(shape=(10,))
dense1 = Dense(64, activation='relu')(input1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1, outputs=output1)

#3. compile and fit
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, batch_size=4, epochs=150, verbose=1, validation_split=0.2)

#4. evaluate and predict
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

# 결과값 ver1
# mse : 3719.33251953125
# mae : 47.651283264160156
# RMSE : 60.986330591424206
# R2 : 0.3942092465679128
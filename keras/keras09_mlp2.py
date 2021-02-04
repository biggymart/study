# 다:1 mlp (3:1)
# 실습 train과 test 분리해서 소스를 완성하시오

# 다:다 mlp (3:3, reshape)
# x1, x2, x3 = 환율, 금리, 국제 유가; y1, y2, y3 = 삼성전자주가, 내일환율, 내일금리;
# 다음의 수식과 같이 모델링 됨: w1x1 + w2x2 + w3x3 + b = y1 + y2 + y3

input_num = 3
output_num = 1

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#1. data
x = np.array([range(100), range(301, 401), range(1, 101)])
y = np.array(range(711,811))
# print(x.shape) # (3, 100)
# print(y.shape) # (100,)

x = np.transpose(x)
# feature가 3개인 것을 의도했으므로 행과 열을 바꿔줌
# print(x.shape) # (100, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66) 
# x의 80행 3열이 x_train으로 됨, random_state로 난수 시드 고정

#2. model
model = Sequential()
model.add(Dense(10, input_dim=input_num)) # Input
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(output_num)) # Output

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100 ,batch_size=1, validation_split=0.2) 
# 각 col 별로 20퍼센트라는 의미 (즉 row가 100개면 그 중 20개), batch_size=1이라는 의미는 shape이 (1,3)인 배열이 한 번에 훈련됨

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

# 결과 (주석 단축키: ctrl + /)
# loss : 2.2351742678949904e-09
# mae : 2.441406286379788e-05
# RMSE : 4.727762873788351e-05
# R2 : 0.9999999999959829

# 잘 나온 이유는 뭘까?
# y = w1x1 + w2x2 + w3x3 + b 에서 w1, w2, w3이 1로 동일하고 bias만 다르기 때문에 모델을 대충 짧게 만들어도 결과가 잘 나올 수밖에 없다
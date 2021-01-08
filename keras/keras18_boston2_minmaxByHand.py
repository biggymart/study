# ver 1의 문제점은?
# 데이터가 훈련에 적합한 값인가?
# 값의 범위가 0에서 700 정도였는데 데이터 값이 0에서 1 사이에 들어가면 참 잘 나올 거 같은데

# 즉, 데이터가 정규화되지 않았음
# 정규화란, 경제에서 redenomation과 비슷한 작업임 (다루기 쉬운 단위로 축소)
# ver 2에서 배울 내용은... 데이터 정규화의 한 방식인 Scaling임 (MinMaxScaler)
# ver 2는 개념을 알아보기 위해 라이브러리를 이용하지 않고 손수 작업해보자
# {x | (x - min) / (Max - min)} ... 0 <= x <= 1

#1. data
import numpy as np
from sklearn.datasets import load_boston

dataset = load_boston()
x = dataset.data
y = dataset.target

# 데이터 전처리 (MinMaxScaler): 통상적으로 성능이 향상됨 **전처리는 반드시 해야함**
x = x / 711. 
# ver 1에서 np.min(x) == 0.0; np.max(x) == 711.0 을 확인함; 
# 위 정보를 활용하여 line 9에 나와있는 수식을 적용하면 됨; *주의* float로 나누어야 함

# ver 3에서 왔다: 앗, x는 여러 column으로 이루어져있기 때문에 각 column에서 최댓값 최솟값이 다르다
# 만약 일률적으로 특정 column의 최댓값과 최솟값으로 x 전체에 계산하면 안 된다!
# 예를 들어 column 1의 최댓값은 300인데 그걸 711.로 나누면 1.0이 되지 않음
# 즉, 우리가 원하는 결과인 "x 범위 0 ~ 1"이 아니게 됨 == 정규화 실패

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

# print(np.max(x), np.min(x)) # 711.0 0.0

# 실습: 모델을 구성하시오
#2. model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense

input1 = Input(shape=(13,))
dense1 = Dense(128, activation='relu')(input1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
dense1 = Dense(64, activation='relu')(dense1)
output1 = Dense(1)(dense1)
model = Model(inputs=input1, outputs=output1)

#3. compile and fit
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(x_train, y_train, batch_size=4, epochs=150, verbose=1, validation_split=0.2)

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

# 결과 ver1
# mse : 20.25518035888672
# mae : 3.4258222579956055
# RMSE : 4.500575586257058
# R2 : 0.7950448684606252

# 결과 ver2
# mse : 17.2949161529541
# mae : 3.2505249977111816
# RMSE : 4.158715759026131
# R2 : 0.8249987492546886 # (향상됨!)
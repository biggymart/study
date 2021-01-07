from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. data
x = np.array(range(1, 101))
y = np.array(range(1, 101))

from sklearn.model_selection import train_test_split
# 이름 자체가 train과 test로 쪼개주는 것, import 되었으니까 함수임
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, shuffle=False)
print(x_train.shape) # (60,) 스칼라가 60개, 1차원
print(x_test.shape) # (40,)
print(y_train.shape) # (60,)
print(y_test.shape) # (40,)

# 데이터 범위의 중요성 (테스트와 훈련의 범위가 유사하게 만들어야 결과 성능이 더 정확함, 그래서 셔플해야 함)
# 순서가 뒤섞인 숫자가 60개 나옴, 순서가 정렬된 숫자보다 더 성능이 좋음
# 다음 달에 일본이랑 축구 경기를 하는데 두 팀으로 나눴고 매일같이 훈련했어, 
# A랑 B랑 나눠서 했는데 매번 A팀이 이겼다고 해서 일본 팀을 이길 수 있을까? 모른다
# 1부터 80까지 정렬된 데이터로 훈련하면 weight값이 정확히 1이 아니라 1.0001 이나 0.9999 정도로 근접할 뿐임. 
# 그래서 만약 x가 10000일 때의 값을 예측하라고 하면 오차가 생길 수 밖에 없음
# 오히려 무작위로 해서 80에서 100 사이에 있는 값도 나올 수 있게 해서 하는 게 더 정확하게 나옴

#2. model
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100)

#4. evaluate and predict
loss, mae = model.evaluate(x_test, y_test)
print('loss :', loss, 'mae :', mae)

y_predict = model.predict(x_test)
print('y_predict :', y_predict)

# 결과치를 주석으로 쓰는 습관을 기르자
# Shuffle=False
# loss : 0.7943292260169983 mae : 0.8442000150680542
# Shuffle=True
# loss : 1.8782408005790785e-05 mae : 0.0036326139234006405
# Shuffle 했을 때 loss가 더 적은 걸 확인할 수 있다.

# 요약
# 1. sklearn.model_selection의 train_test_split 용법
# 2. shuffle 옵션의 중요성
# 다:1 mlp

# 퍼셉트론에 대해서 검색해봐라
# 가장 간단한 형태의 피드포워드(Feedforward) 네트워크 - 선형분류기
# 퍼셉트론이 동작하는 방식: 
# 각 노드의 가중치와 입력치를 곱한 것을 모두 합한 값이 활성함수에 의해 판단되는데, 
# 그 값이 임계치(보통 0)보다 크면 뉴런이 활성화되고 결과값으로 1을 출력한다. 뉴런이 활성화되지 않으면 결과값으로 -1을 출력한다. 
# 우리가 봐왔던 인공신경망은 MLP (Multi-Layer Perceptron)

import numpy as np

#1. data
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],[11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x = np.transpose(x)
print(x)
print(x.shape) # (10,2)

# print(x)
# xx = np.arraxy([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# print(x.shape) # (10,) 10 scalars

# 
# print(x.shape)x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
# print(x.shape) # (2, 10) 2 scalars 10 dimension (input_dim=3)

# 엑셀을 다들 써봤을 거야. 예를 들어 성적 분포라고 하면, 키워드는 1st row (columns)에 놓고 아래 rows에 각 관측치를 넣잖아.
# 모델을 짜는 데 "행무시, 열우선", 행은 batch_size 및 epoch에 쓰이고
# 열 == 피처; keywords go into columns --> columns first; column is also called as "feature" (feature importance, feature engineering etc.)
# 10년 데이터, x <- (온도, 습도, 미세먼지농도), y <- (삼성전자 주가), y = w1x1 + w2x2 + w3x3 + b, 훈련 셋 : 테스트 셋 = 7 : 3
# 10년 데이터, x <- (온도, 습도, 미세먼지농도, 국제유가), y <- (날씨), 결과는 돌려봐야하겠지만 국제유가는 날씨 예측에 별 영향력이 없을 것 같다는 느낌이 들죠, feature importance의 개념
# 히든 레이어를 알 수 없다? 아니다, 알 수 있다. 작은 신경망 모델의 경우 각 파라미터는 단순연산이므로 웨이트를 다음 노드로 넘겨주는 과정을 되풀이 함. 이 과정을 역산하면 히든 레이어를 알 수 있음.

# 다음 행렬의 shape을 구하시오.      (폭 * 높이 * 가로)
# 1. [[1, 2, 3], [4, 5, 6]]         2 by 3
# 2. [[1,2],[3,4],[5,6]]            3 by 2
# 3. [[[1,2,3],[4,5,6]]]            1 by 2 by 3
# 4. [[1,2,3,4,5,6]]                1 by 6
# 5. [[[1,2],[3,4]],[[5,6],[7,8]]]  2 by 2 by 2
# 6. [[1],[2],[3]]                  3 by 1
# 7. [[[1],[2]],[[3],[4]]]          2 by 2 by 1

#2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense # 텐서플로 2.0부터 케라스 흡수함
# from keras.layers import Dense 벡엔드에서 케라스를 깔고 실행하기 때문에 좀 느림

model = Sequential()
model.add(Dense(10, input_dim=2)) # 이 파일은 결국 input_dim=2 위해 만든 것임
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. compile and fit
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, batch_size=1, validation_split=0.2) # 각 col 별로 20퍼센트라는 의미

#4. evaluate and predict
loss, mae = model.evaluate(x, y)
print( 'loss :', loss, "\nmae :", mae)

y_predict = model.predict(x)
# print(y_predict)
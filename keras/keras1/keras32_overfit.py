# 과적합: 어떠한 훈련 데이터에 대해서 모델을 완벽하게 맞게 해서 만들면 나중에 다른 데이터가 들어오면 쓸모가 없음
# test/validation에서 조금만 달라져도 정확도가 낮아짐, 훈련할 때 버릴 애들은 버리는 게 낫다

# 그렇다면 과적합이 되지 않게 하려면 어떻게 해야 할까
# 1. 훈련 데이터를 늘린다
# 2. feature를 줄인다
#   (feature가 많다는 건 y = w1x1 + w2x2 + w3x3 + ... 이렇게 연산이 많아짐
#   버릴 특성은 버린다, mnist할 때 600여개 feature이 있는데 100개 정도로 줄여야 한다)
# 3. regularization (정규화) 실시
# 4. Dropout (DL 됨, ML 안 됨)
#   Dropout: 노드를 제거하는 게 아니라 훈련 때 사용하지 않는 것 
#   하이퍼파라미터 자동화할 때 편함
#   summary 확인해보면 total param # 같음 (train에는 적용하지 않지만 확보한 메모리 양은 같음, test할 때는 원래 노드 다 활용)
#   어떻게 보면 max 값을 설정해놓는다는 개념과 비슷함
# 5. Ensemble
#   카더라 통신이긴 한데 통상 2~5% 향상이 있다고 하는 놈들이 있다

''' 자동화 예시
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

a = [0.2, 0.3, 0.4]
b = [0.2, 0.3, 0.4]
c = [100, 200, 300]
d = ['relu', 'linear', 'elu', 'selu', 'tanh']

model = Sequential()
model.add(Dense(128, input_dim=13))
model.add(Dropout(a))
model.add(Dense(c, activation=d))
model.add(Dropout(b))
model.add(Dense(c, activation=d))
model.add(Dropout(a))
model.add(Dense(c, activation=d))
model.add(Dropout(b))
model.add(Dense(1))
'''
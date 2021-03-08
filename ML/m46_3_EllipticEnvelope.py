from sklearn.covariance import EllipticEnvelope
import numpy as np
# aaa = np.array([[1,2,-10000,3,4,6,7,8,90,100,5000]])
aaa = np.transpose(aaa)
# sklearn는 대부분 벡터로 인풋을 받음

outlier = EllipticEnvelope(contamination=.1)
# "outlier가 10퍼센트 있다고 간주하고 이상치를 찾아라"
# 통상적으로 10퍼센트 미만으로 설정함
# 가우스 분포와 공분산을 사용하여 처리하는 것이 차이 
outlier.fit(aaa)

print(outlier.predict(aaa))


# contamination=.3
# [ 1  1 -1  1  1  1  1  1  1 -1 -1]
# contamination=.2
# [ 1  1 -1  1  1  1  1  1  1  1 -1]
# contamination=.1
# [ 1  1 -1  1  1  1  1  1  1  1  1]

# ======================
# 1차원 말고 2차원도 될까?
# 응, 된다
# 기준은 열

# aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],
#                 [1000,2000,3,4000,5000,6000,7000,8,9000,10000]])

# [ 1  1  1  1 -1  1  1  1  1  1]
# "(10, 2) 크기의 데이터 전체를 봤을 때 5 row에 어딘가 이상치가 있다"
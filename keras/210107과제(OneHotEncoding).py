import numpy as np

data = np.array([3,6,5,4,2])

#1. tensorflow.keras의 to_categorical
from tensorflow.keras.utils import to_categorical
kr_ohe = to_categorical(data)
print(kr_ohe)
# [[0. 0. 0. 1. 0. 0. 0.] 
#  [0. 0. 0. 0. 0. 0. 1.] 
#  [0. 0. 0. 0. 0. 1. 0.] 
#  [0. 0. 0. 0. 1. 0. 0.] 
#  [0. 0. 1. 0. 0. 0. 0.]]
''' keras는 0부터 index가 시작해야 하기 때문에 0, 1 요소가 빠져있는 data를 확인하고, 0과 1 index를 만들어준다
(시작점은 반드시 0부터 해야하는 고집 쎈 녀석)'''
print(kr_ohe.shape)
# (5, 7)
''' 따라서 2 feature이 자동적으로 추가되어 column은 7이다'''

#2. sklearn의 OneHotEncoder
from sklearn.preprocessing import OneHotEncoder #LabelEncoder (만약 y 데이터가 0, 1 ,2 등 이런 식으로 분류되어 있지 않으면 써야함)
enc = OneHotEncoder()
sk_ohe = enc.fit_transform(data.reshape(-1,1)).toarray()
print(sk_ohe)
#[[0. 1. 0. 0. 0.] 
#  [0. 0. 0. 0. 1.] 
#  [0. 0. 0. 1. 0.] 
#  [0. 0. 1. 0. 0.] 
#  [1. 0. 0. 0. 0.]]
''' sklearn은 있는 것만 따져서 one-hot encoding scheme을 만들어준다
(기준점이야 있는 애들로 만들면 되지하는 유두리 있는 놈)'''
print(sk_ohe.shape)
# (5, 5)
''' 따라서 다른 column을 추가하지 않는다 ==> 5 col'''

# p.s> sklearn 같은 경우, method가 좀 화려한 느낌이라서 헷갈리면 아래 링크를 보도록 하자
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
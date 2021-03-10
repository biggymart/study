# y 레이블에 대해서 조정할 수 있는 권한이 있다고 하면 어떨까?
# 예를 들어 개의 품종까지 분류하는 것이 아니라 개 혹은 고양이만 분류하면
# 성능이 더 좋은 모델을 뽑을 수 있다

# 기존의 와인 데이터셋에서는 레이블이 0, 1, 2 만 있었다

import pandas as pd
import numpy as np

wine = pd.read_csv('../data/csv/winequality-white.csv',
    sep=';', header=0, index_col=None
)
print(wine.head())
print(wine.shape) # (4898, 12)
print(wine.describe())

wine_npy = wine.values

x = wine_npy[:, :11]
y = wine_npy[:, 11]

print(x.shape, y.shape) # (4898, 11) (4898,)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, shuffle=True, train_size=0.8
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape)
# (3918, 11) (980, 11)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# model = KNeighborsClassifier() # score : 0.5663265306122449
model = RandomForestClassifier() # score : 0.713265306122449
# model = XGBClassifier() # score : 0.6816326530612244

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("score :", score)






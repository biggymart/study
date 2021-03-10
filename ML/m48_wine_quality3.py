import pandas as pd
import numpy as np

wine = pd.read_csv('../data/csv/winequality-white.csv',
    sep=';', header=0, index_col=None
)
print(wine.head())
print(wine.shape) # (4898, 12)
print(wine.describe())

wine_npy = wine.values

# x = wine_npy[:, :11]
# y = wine_npy[:, 11]

y = wine['quality']
x = wine.drop('quality', axis=1)

newlist = []
for i in list(y):
    if i <= 4:
        newlist += [0] # 0등급
    elif i <= 7:
        newlist += [1] # 1등급
    else:
        newlist += [2] # 2등급
y = np.array(newlist)



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
model = RandomForestClassifier() # score : 0.713265306122449 -> 0.9459183673469388
# model = XGBClassifier() # score : 0.6816326530612244

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print("score :", score)





